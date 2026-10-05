import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from drift_detection import fetch_full_history_returns
from model_registry import get_next_model_version, save_model_to_storage, load_model_from_storage

MODEL_TYPE = "lasso"
FILENAME_PREFIX = "Lasso"
LASSO_MAX_ITER = 10000


class LassoModel:
    """
    Regresja liniowa z regularyzacją L1 (Lasso) na dziennych stopach zwrotu,
    nie na poziomach cen - ceny są niestacjonarne, więc model na surowych
    poziomach byłby bez sensu statystycznego. Selekcja cech odbywa się
    automatycznie przez regularyzację (nieistotne cechy dostają współczynnik
    0), siła regularyzacji (alpha) dobierana przez cross-walidację (LassoCV).

    Cechy są standaryzowane przed dopasowaniem - kara L1 zależy od skali
    cechy, więc bez standaryzacji instrumenty o dużej dziennej zmienności
    (krypto, gaz) potrzebowałyby mniejszego współczynnika na ten sam efekt
    i byłyby karane słabiej niż spokojne (obligacje, waluty), niezależnie od
    swojej przydatności. Skaler jest dopasowywany w każdym foldzie
    TimeSeriesSplit osobno (pipeline), więc nie widzi danych walidacyjnych.
    """

    def __init__(self, storage_path: str | None = None, baseline_stats: dict | None = None,
                 config: dict | None = None):
        """Jeśli podano storage_path, od razu wczytuje wcześniej wytrenowany
        model z Supabase Storage."""
        self.storage_path = storage_path
        self.today_data: dict | None = None
        self.model = None
        self.selected_features: list[str] = []
        self.baseline_stats = baseline_stats
        # config: {"cv_splits":.., ...} - z pipeline_config.
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
        actual_y z kolejnego dnia), i wybiera cechy przez cross-walidowaną
        regularyzację L1 (LassoCV). Nic nie robi, jeśli retrain_required=False.
        """
        if not retrain_required:
            return

        print("LassoModel: rozpoczynam retrening...")
        returns_df, dead_features = fetch_full_history_returns(
            as_of_date, window_years=int(self.config["training_window_years"])
        )

        if dead_features:
            print(f"LassoModel: wykluczam z kandydatów (brak danych w ostatnich 10 dniach): {dead_features}")
            returns_df = returns_df.drop(columns=dead_features)

        train_df = returns_df.copy()
        train_df["__target__"] = returns_df["actual_y"].shift(-1)

        rows_before = len(train_df)
        train_df = train_df.dropna()
        print(f"LassoModel: trening na {len(train_df)} wierszach "
              f"(odrzucono {rows_before - len(train_df)} wierszy z brakami danych).")

        self.train_start_date = train_df.index.min().strftime("%Y-%m-%d")
        self.train_end_date = train_df.index.max().strftime("%Y-%m-%d")

        y_train = train_df.pop("__target__")
        X_train = train_df
        all_candidates = list(X_train.columns)

        cv_pipeline = make_pipeline(
            StandardScaler(),
            LassoCV(cv=TimeSeriesSplit(n_splits=int(self.config["cv_splits"])), max_iter=LASSO_MAX_ITER),
        ).fit(X_train, y_train)
        cv_model = cv_pipeline[-1]
        selected_features = [f for f, coef in zip(all_candidates, cv_model.coef_) if coef != 0]

        if not selected_features:
            # Regularyzacja wyzerowała WSZYSTKIE współczynniki - walidacja
            # wskazała, że żadna cecha nie przewiduje jutrzejszego zwrotu lepiej
            # niż sama średnia. To uczciwy wynik, nie błąd: model przewiduje
            # wtedy średni zwrot z okna treningowego (bez predyktorów).
            print("LassoModel: WSZYSTKIE współczynniki wyzerowane przez regularyzację - "
                  "model bez predyktorów, prognoza = średni zwrot z okna treningowego.")
            final_model = DummyRegressor(strategy="mean").fit(X_train, y_train)
        else:
            # Refit na samych wybranych cechach (najpierw wybierz, potem dopasuj
            # finalny, mniejszy model) - upraszcza predict()/SHAP, bo nie trzeba
            # karmić modelu wszystkimi kandydatami co dzień.
            final_model = make_pipeline(
                StandardScaler(), Lasso(alpha=cv_model.alpha_, max_iter=LASSO_MAX_ITER)
            ).fit(X_train[selected_features], y_train)

        self.model = final_model
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

        print(f"LassoModel: wytrenowano wersję v{self.model_version} (alpha={cv_model.alpha_:.6f}), "
              f"wybrane cechy: {self.selected_features}")

    def predict(self) -> dict:
        """Predykcja jutrzejszej ceny złota na podstawie dzisiejszych
        zwrotów wybranych cech."""
        if self.today_data is None:
            raise ValueError("Brak danych - wywołaj load_data() przed predict().")
        if self.model is None:
            raise ValueError("Brak wytrenowanego modelu - wywołaj retrain_if_needed(True, ...) przed predict().")

        last_actual_y_level = self.today_data["last_actual_y_level"]

        if not self.selected_features:
            # Model bez predyktorów (patrz retrain_if_needed) - stały,
            # średni zwrot, brak cech do wytłumaczenia przez SHAP.
            predicted_return = float(np.ravel(self.model.constant_)[0])
            shap_values = {}
        else:
            returns = self.today_data["returns"]
            x_values = [returns[feature] for feature in self.selected_features]
            x_row = pd.DataFrame([x_values], columns=self.selected_features)
            predicted_return = float(self.model.predict(x_row)[0])
            shap_values = self._compute_shap(x_row, last_actual_y_level)

        predicted_price = last_actual_y_level * (1 + predicted_return)

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

    def _compute_shap(self, x_row: pd.DataFrame, last_actual_y_level: float) -> dict:
        """SHAP dla modelu liniowego - shap.LinearExplainer z tłem = średnie
        z baseline_stats (matematycznie poprawny punkt odniesienia dla
        modelu liniowego).

        Model przewiduje ZWROT, nie cenę, więc surowy SHAP jest w
        jednostkach zwrotu (ułamek) - przemnożenie przez last_actual_y_level
        daje wkład w dolarach."""
        means = np.array([[self.baseline_stats[f]["mean"] for f in self.selected_features]])
        coef, intercept = self._original_scale_coefficients()

        explainer = shap.LinearExplainer((coef, intercept), means)
        raw_shap_values = explainer.shap_values(x_row.values)[0]

        return {
            feature: float(value) * last_actual_y_level
            for feature, value in zip(self.selected_features, raw_shap_values)
        }

    def _original_scale_coefficients(self) -> tuple[np.ndarray, float]:
        """Współczynniki w jednostkach SUROWYCH zwrotów (nie standaryzowanych),
        żeby SHAP liczył się na tych samych danych co zawsze: beta = beta_std /
        skala, wyraz wolny przesunięty o średnie skalera. Obsługuje też starsze
        wersje zapisane jako sam Lasso, bez skalera."""
        if not isinstance(self.model, Pipeline):
            return self.model.coef_, float(self.model.intercept_)
        scaler, lasso = self.model[0], self.model[-1]
        coef = lasso.coef_ / scaler.scale_
        intercept = float(lasso.intercept_ - np.sum(lasso.coef_ * scaler.mean_ / scaler.scale_))
        return coef, intercept
