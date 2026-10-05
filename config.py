from db_client import supabase

# {key: (wartość, opis)} - używane WYŁĄCZNIE jako wartości inicjujące, jeśli
# pipeline_config nie ma jeszcze żadnego aktywnego wiersza (pierwsze
# uruchomienie systemu) - ten sam wzorzec bootstrapu co przy models_logs
# ("brak wpisu -> wymuszam pierwszy trening, retrain_trigger='init'").
DEFAULT_CONFIG = {
    "data_drift_z_threshold": (3.0, "Próg Z-score dla Data Drift"),
    "data_drift_pct_threshold": (0.20, "Odsetek zdryfowanych cech wymagany do retreningu"),
    "concept_drift_delta": (
        0.0005, "Page-Hinkley delta (ułamek ceny, na błędzie względem modelu naiwnego)"
    ),
    "concept_drift_lambda": (
        0.03, "Page-Hinkley lambda_threshold (ułamek ceny, na błędzie względem modelu naiwnego)"
    ),
    "active_model_margin": (0.005, "Margines względnej poprawy MAPE do przełączenia"),
    "active_model_streak_days": (2, "Ile dni z rzędu trzeba pobijać aktywny model"),
    "rolling_mape_window_days": (29, "Okno kroczącego MAPE (dni)"),
    "training_window_years": (
        3, "Długość okna historii używanej do treningu OLS/Lasso/RF/XGBoost (lata)"
    ),
    "ols_p_value_threshold": (0.05, "Próg eliminacji wstecznej OLS"),
    "ols_vif_threshold": (10.0, "Próg VIF"),
    "tree_cumulative_importance_threshold": (0.90, "Próg skumulowanej ważności cech (RF/XGBoost)"),
    "cv_splits": (5, "Liczba foldów TimeSeriesSplit"),
    "rf_param_grid": (
        {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2, 4]},
        "Siatka GridSearchCV - Random Forest",
    ),
    "xgboost_param_grid": (
        {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
        "Siatka GridSearchCV - XGBoost",
    ),
}


def load_config() -> dict:
    """
    Wczytuje wszystkie AKTYWNE wartości z pipeline_config (progi, alfy,
    siatki hiperparametrów) - wołane RAZ na starcie main(), wynik przekazywany
    dalej jako jeden obiekt (nie osobne argumenty per funkcja), żeby cały
    przebieg działał na spójnym zestawie ustawień, nawet gdyby ktoś zmienił
    coś w trakcie przez Streamlit.

    Jeśli brak jakichkolwiek aktywnych wpisów (pierwsze uruchomienie systemu),
    zapisuje DEFAULT_CONFIG do bazy jako pierwsze, aktywne wiersze - nie
    tylko cichy fallback w pamięci, tylko realny ślad w historii configu od
    samego początku.

    Historia zmian (nieaktywne wiersze) zostaje w tabeli, ale load_config()
    jej nie zwraca.
    """
    response = (
        supabase.table("pipeline_config")
        .select("key, value")
        .eq("active", True)
        .execute()
    )
    rows = response.data

    if not rows:
        print("Brak jakichkolwiek aktywnych wpisów w pipeline_config - zakładam "
              "pierwsze uruchomienie, zapisuję wartości domyślne jako inicjujące.")
        payload = [
            {"key": key, "value": value, "description": description, "active": True}
            for key, (value, description) in DEFAULT_CONFIG.items()
        ]
        supabase.table("pipeline_config").insert(payload).execute()
        return {key: value for key, (value, _description) in DEFAULT_CONFIG.items()}

    return {row["key"]: row["value"] for row in rows}
