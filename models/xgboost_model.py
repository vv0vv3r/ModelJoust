import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from drift_detection import fetch_full_history_returns
from model_registry import get_next_model_version, save_model_to_storage, load_model_from_storage

MODEL_TYPE = "xgboost"
FILENAME_PREFIX = "XGBoost"
RANDOM_STATE = 19
N_JOBS = -1  # argument pozwalający, by kod używał wszystkich dostępnych rdzeni CPU


class XGBoostModel:
    """
    XGBoost na dziennych stopach zwrotu, nie na poziomach cen - ceny są
    niestacjonarne, więc model na surowych poziomach byłby bez sensu
    statystycznego. Selekcja cech: twardy próg na skumulowanej ważności
    cech (feature_importances_) - bierze się tyle najważniejszych cech, ile
    trzeba, żeby zebrać tree_cumulative_importance_threshold łącznej
    ważności. SHAP liczony przez TreeExplainer.

    Hiperparametry dobierane przez GridSearchCV na siatce z
    config["xgboost_param_grid"], z cv=TimeSeriesSplit zamiast domyślnego,
    losowo tasującego KFold - dla szeregu czasowego zwykły KFold pozwoliłby
    foldowi walidacyjnemu zawierać dane sprzed foldu treningowego
    (lookahead bias).
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
        # config: {"tree_cumulative_importance_threshold":.., "cv_splits":..,
        # "xgboost_param_grid":..} - z pipeline_config.
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
        actual_y z kolejnego dnia), dobiera hiperparametry przez
        GridSearchCV i wybiera cechy po skumulowanej ważności. Nic nie
        robi, jeśli retrain_required=False.
        """
        if not retrain_required:
            return

        print("XGBoostModel: rozpoczynam retrening...")
        returns_df, dead_features = fetch_full_history_returns(
            as_of_date, window_years=float(self.config["training_window_years"])
        )

        if dead_features:
            print(f"XGBoostModel: wykluczam z kandydatów (brak danych w ostatnich 10 dniach): {dead_features}")
            returns_df = returns_df.drop(columns=dead_features)

        train_df = returns_df.copy()
        train_df["__target__"] = returns_df["actual_y"].shift(-1)

        rows_before = len(train_df)
        train_df = train_df.dropna()
        print(f"XGBoostModel: trening na {len(train_df)} wierszach "
              f"(odrzucono {rows_before - len(train_df)} wierszy z brakami danych).")

        self.train_start_date = train_df.index.min().strftime("%Y-%m-%d")
        self.train_end_date = train_df.index.max().strftime("%Y-%m-%d")

        y_train = train_df.pop("__target__")
        X_train = train_df
        all_candidates = list(X_train.columns)

        grid_search = GridSearchCV(
            XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE),  # bez n_jobs - unika zagnieżdżonej równoległości z GridSearchCV
            param_grid=self.config["xgboost_param_grid"],
            cv=TimeSeriesSplit(n_splits=int(self.config["cv_splits"])),
            scoring="neg_mean_squared_error",
            n_jobs=N_JOBS,
        )
        grid_search.fit(X_train, y_train)
        full_model = grid_search.best_estimator_
        print(f"XGBoostModel: najlepsze hiperparametry (GridSearchCV): "
              f"{grid_search.best_params_}")

        # Sortuj cechy wg ważności malejąco, bierz tyle ile trzeba, żeby
        # zebrać tree_cumulative_importance_threshold łącznej ważności.
        order = np.argsort(full_model.feature_importances_)[::-1]
        cum_importance = np.cumsum(full_model.feature_importances_[order])
        cutoff_idx = int(np.searchsorted(cum_importance, self.config["tree_cumulative_importance_threshold"])) + 1
        selected_features = [all_candidates[i] for i in order[:cutoff_idx]]

        # Refit na samych wybranych cechach, tymi samymi hiperparametrami co
        # znalezione wyżej - bez powtarzania całego GridSearchCV na
        # mniejszym zbiorze cech.
        final_model = XGBRegressor(
            **grid_search.best_params_, objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=N_JOBS
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

        print(f"XGBoostModel: wytrenowano wersję v{self.model_version}, "
              f"wybrane cechy ({len(self.selected_features)}/{len(all_candidates)} kandydatów): "
              f"{self.selected_features}")

    def predict(self) -> dict:
        """Predykcja jutrzejszej ceny złota na podstawie dzisiejszych
        zwrotów wybranych cech."""
        if self.today_data is None:
            raise ValueError("Brak danych - wywołaj load_data() przed predict().")
        if self.model is None or not self.selected_features:
            raise ValueError("Brak wytrenowanego modelu - wywołaj retrain_if_needed(True, ...) przed predict().")

        returns = self.today_data["returns"]
        x_values = [returns[feature] for feature in self.selected_features]

        x_row = pd.DataFrame([x_values], columns=self.selected_features)
        predicted_return = float(self.model.predict(x_row)[0])

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

    def _compute_shap(self, x_row: pd.DataFrame, last_actual_y_level: float) -> dict:
        """SHAP dla modelu drzewiastego - TreeExplainer (dokładny, wspiera
        XGBoost natywnie, bez ręcznie liczonego tła jak LinearExplainer).

        Model przewiduje ZWROT, nie cenę, więc surowy SHAP jest w
        jednostkach zwrotu (ułamek) - przemnożenie przez last_actual_y_level
        daje wkład w dolarach."""
        explainer = shap.TreeExplainer(self.model)
        raw_shap_values = explainer.shap_values(x_row)

        return {
            feature: float(value) * last_actual_y_level
            for feature, value in zip(self.selected_features, raw_shap_values[0])
        }
