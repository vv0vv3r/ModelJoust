import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from drift_detection import fetch_full_history_returns
from model_registry import get_next_model_version, save_model_to_storage, load_model_from_storage

MODEL_TYPE = "OLS"
FILENAME_PREFIX = "OLS"


class OLSModel:
    """
    Regresja liniowa (OLS) na dziennych stopach zwrotu, nie na poziomach cen
    - ceny są niestacjonarne, więc model na surowych poziomach byłby bez
    sensu statystycznego. Cechy dobierane automatycznie: eliminacja wsteczna
    po p-value, potem filtr współliniowości (VIF). "actual_y" (wczorajszy
    zwrot złota) jest jednym z kandydatów jako człon autoregresyjny.
    """

    def __init__(self, storage_path: str | None = None, baseline_stats: dict | None = None,
                 config: dict | None = None):
        """Jeśli podano storage_path, od razu wczytuje wcześniej wytrenowany
        model z Supabase Storage."""
        self.storage_path = storage_path
        self.today_data: dict | None = None
        self.model = None
        self.selected_features: list[str] = []
        # baseline_stats: jedyne źródło prawdy to models_logs, skąd
        # main_pipeline.py je pobiera. Potrzebne przy KAŻDYM predict() jako
        # tło dla SHAP, nie tylko w dniu retreningu.
        self.baseline_stats = baseline_stats
        # config: {"ols_p_value_threshold":.., "ols_vif_threshold":.., ...}
        # z pipeline_config, wczytane raz na starcie main().
        self.config = config
        self.model_version: int | None = None
        self.train_start_date: str | None = None
        self.train_end_date: str | None = None

        if storage_path:
            bundle = load_model_from_storage(storage_path)
            self.model = bundle["model"]
            self.selected_features = bundle["selected_features"]

    def load_data(self, today_data: dict) -> None:
        """today_data = {"returns": {...}, "last_actual_y_level": float}."""
        self.today_data = today_data

    def retrain_if_needed(self, retrain_required: bool, as_of_date: pd.Timestamp) -> None:
        """
        Pełny retrening: pobiera całą historię zwrotów, wyklucza cechy
        martwe w ostatnich 10 dniach, buduje zbiór treningowy (target =
        actual_y z kolejnego dnia), i wybiera cechy eliminacją wsteczną
        (p-value) z filtrem VIF. Nic nie robi, jeśli retrain_required=False.
        """
        if not retrain_required:
            return

        print("OLSModel: rozpoczynam retrening...")
        returns_df, dead_features = fetch_full_history_returns(
            as_of_date, window_years=int(self.config["training_window_years"])
        )

        if dead_features:
            print(f"OLSModel: wykluczam z kandydatów (brak danych w ostatnich 10 dniach): {dead_features}")
            returns_df = returns_df.drop(columns=dead_features)

        train_df = returns_df.copy()
        train_df["__target__"] = returns_df["actual_y"].shift(-1)

        rows_before = len(train_df)
        train_df = train_df.dropna()
        print(f"OLSModel: trening na {len(train_df)} wierszach "
              f"(odrzucono {rows_before - len(train_df)} wierszy z brakami danych).")

        self.train_start_date = train_df.index.min().strftime("%Y-%m-%d")
        self.train_end_date = train_df.index.max().strftime("%Y-%m-%d")

        y_train = train_df.pop("__target__")
        X_train = train_df

        model, selected_features = self._backward_elimination(X_train, y_train)
        model, selected_features = self._filter_vif(X_train[selected_features], y_train)

        self.model = model
        self.selected_features = selected_features
        # baseline_stats liczone dla WSZYSTKICH cech kandydujących, nie tylko
        # wybranych - compute_feature_drift_flags sprawdza drift dla całego
        # zbioru, dopiero potem filtrowanego do selected_features.
        self.baseline_stats = {
            col: {"mean": float(returns_df[col].mean()), "std": float(returns_df[col].std())}
            for col in returns_df.columns
        }
        self.model_version = get_next_model_version(MODEL_TYPE)
        self.storage_path = save_model_to_storage(self.model, self.selected_features, FILENAME_PREFIX, self.model_version)

        print(f"OLSModel: wytrenowano wersję v{self.model_version}, "
              f"wybrane cechy: {self.selected_features}")

    def predict(self) -> dict:
        """Predykcja jutrzejszej ceny złota na podstawie dzisiejszych
        zwrotów wybranych cech."""
        if self.today_data is None:
            raise ValueError("Brak danych - wywołaj load_data() przed predict().")
        if self.model is None or not self.selected_features:
            raise ValueError("Brak wytrenowanego modelu - wywołaj retrain_if_needed(True) przed predict().")

        returns = self.today_data["returns"]
        x_values = [returns[feature] for feature in self.selected_features]

        x_row = pd.DataFrame([x_values], columns=self.selected_features)
        x_row_with_const = sm.add_constant(x_row, has_constant="add")

        predicted_return = float(self.model.predict(x_row_with_const)[0])

        last_actual_y_level = self.today_data["last_actual_y_level"]
        predicted_price = last_actual_y_level * (1 + predicted_return)

        shap_values = self._compute_shap(x_row, last_actual_y_level)

        return {
            "predicted_value": predicted_price,
            "shap_values": shap_values,
            "selected_features": self.selected_features,
            "baseline_stats": self.baseline_stats,
            "storage_path": self.storage_path,
            "model_version": self.model_version,
            "train_start_date": self.train_start_date,
            "train_end_date": self.train_end_date,
        }

    # --- Wewnętrzne pomocnicze ---

    def _backward_elimination(self, X: pd.DataFrame, y: pd.Series):
        """Iteracyjnie usuwa cechę z najwyższym p-value, dopóki wszystkie
        pozostałe są istotne (p <= ols_p_value_threshold) albo zostanie jedna.
        P-value z błędów standardowych HAC (patrz _fit_hac)."""
        p_value_threshold = self.config["ols_p_value_threshold"]
        features = list(X.columns)
        while True:
            X_with_const = sm.add_constant(X[features])
            model = _fit_hac(y, X_with_const)
            p_values = model.pvalues.drop("const")
            worst_feature = p_values.idxmax()
            worst_p = p_values.max()

            if worst_p <= p_value_threshold or len(features) == 1:
                return model, features

            features.remove(worst_feature)

    def _filter_vif(self, X: pd.DataFrame, y: pd.Series):
        """Iteracyjnie usuwa cechę z najwyższym VIF, dopóki wszystkie
        pozostałe są poniżej ols_vif_threshold albo zostanie jedna."""
        vif_threshold = self.config["ols_vif_threshold"]
        features = list(X.columns)
        while len(features) > 1:
            X_with_const = sm.add_constant(X[features])
            vif_values = pd.Series(
                [variance_inflation_factor(X_with_const.values, i)
                 for i in range(1, X_with_const.shape[1])],
                index=features,
            )
            worst_feature = vif_values.idxmax()
            worst_vif = vif_values.max()

            if worst_vif <= vif_threshold:
                break
            features.remove(worst_feature)

        X_with_const = sm.add_constant(X[features])
        model = _fit_hac(y, X_with_const)
        return model, features

    def _compute_shap(self, x_row: pd.DataFrame, last_actual_y_level: float) -> dict:
        """SHAP dla modelu liniowego - shap.LinearExplainer z tłem = średnie
        z baseline_stats (przybliżenie bez trzymania pełnych danych
        treningowych w pamięci).

        Przemnożenie przez last_actual_y_level daje przybliżony wkład w
        dolarach, zgodny z sumą SHAP."""
        means = np.array([[self.baseline_stats[f]["mean"] for f in self.selected_features]])
        coef = self.model.params.drop("const").reindex(self.selected_features).values
        intercept = self.model.params["const"]

        explainer = shap.LinearExplainer((coef, intercept), means)
        raw_shap_values = explainer.shap_values(x_row.values)[0]

        return {
            feature: float(value) * last_actual_y_level
            for feature, value in zip(self.selected_features, raw_shap_values)
        }


def _fit_hac(y: pd.Series, X_with_const: pd.DataFrame):
    """OLS z błędami standardowymi HAC (Newey-West). Współczynniki identyczne
    jak w zwykłym OLS - zmieniają się tylko błędy standardowe i p-value.
    Zwykły wzór zakłada stałą wariancję reszt i brak ich autokorelacji, a
    dzienne zwroty mają skupiska zmienności (spokojne i nerwowe okresy), więc
    zwykłe p-value są zaniżone i eliminacja wsteczna zostawiałaby cechy
    dopasowane do szumu. Liczba opóźnień wg reguły Neweya-Westa:
    floor(4 * (T/100)^(2/9)), dla ~3 lat danych = 6."""
    maxlags = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
    return sm.OLS(y, X_with_const).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
