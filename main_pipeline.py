import warnings

import pandas as pd

# Wycisza wyłącznie ResourceWarning z wewnętrznego cache'u JIT numba
# (zależność shap) - nieszkodliwe "unclosed database in sqlite3.Connection",
# zaśmiecające logi tak, że wyglądają jak prawdziwy błąd. Prawdziwe wyjątki
# (tracebacki) nie są tą kategorią, więc nadal będą widoczne normalnie.
warnings.filterwarnings("ignore", category=ResourceWarning)

from data_download import save_latest_market_data
from prediction_evaluation import evaluate_predictions_and_update_system_logs
from drift_detection import fetch_recent_window, build_today_return_row, fetch_active_models_info, compute_feature_drift_flags, check_model_data_drift, update_page_hinkley, compute_dead_features
from models.naive_model import NaiveModel
from models.ols_model import OLSModel
from models.lasso_model import LassoModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from llm_comment import build_llm_comment
from model_registry import save_model_version_if_needed, save_prediction, update_concept_drift_stats
from config import load_config

# Rejestr blackboxów per model_type, rozszerzany o każdy kolejny model.
MODEL_CLASSES = {
    "naive": NaiveModel,
    "OLS": OLSModel,
    "lasso": LassoModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}

def main(as_of_date: pd.Timestamp | None = None):
    """
    Codzienny przebieg pipeline'u: pobranie dzisiejszych danych rynkowych,
    ewaluacja wczorajszych predykcji, sprawdzenie Data Driftu, Concept
    Driftu i dostępności danych per model (decyzja o retreningu), a na
    końcu wygenerowanie i zapisanie predykcji na kolejny dzień sesyjny dla
    każdego blackboxa.

    as_of_date=None -> uruchomienie na dziś (wg czasu US/Eastern). Podanie
    konkretnej daty -> uzupełnienie zaległego dnia lub symulacja historyczna,
    reszta funkcji działa identycznie w obu przypadkach.
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    today_str = as_of_date.strftime("%Y-%m-%d")

    # Odczyt RAZ na starcie - cały przebieg działa na jednym, spójnym
    # zestawie progów/hiperparametrów, nawet gdyby ktoś zmienił coś w
    # trakcie przez Streamlit.
    config = load_config()

    save_latest_market_data(as_of_date)
    todays_errors = evaluate_predictions_and_update_system_logs(today_str, config)

    window_df = fetch_recent_window(as_of_date)
    today_data = build_today_return_row(window_df)  # liczone RAZ, reużywane niżej i przy predykcji

    dead_features = compute_dead_features(window_df)
    if dead_features:
        print(f"Cechy bez żadnej wartości w ostatnich 10 dniach (yfinance przestało je raportować): {dead_features}")

    models_logs_by_name = fetch_active_models_info()

    # Dzisiejszy błąd modelu naiwnego = wielkość dzisiejszego ruchu ceny złota
    # ("jak trudny był dzień"). Concept Drift śledzi błąd KAŻDEGO modelu
    # pomniejszony o tę wartość - sam błąd bezwzględny rośnie też przy wzroście
    # zmienności rynku (dla wszystkich modeli naraz, łącznie z naiwnym), a to
    # nie jest zmiana relacji cechy->cena. Dla samego naiwnego sygnał jest
    # zawsze 0, więc detektor nigdy się dla niego nie uruchomi.
    naive_error = todays_errors.get("naive")

    # --- Krok 1: dla każdego modelu ustal, czy potrzebny jest retrening
    # (brak wpisu w models_logs, martwa cecha, Data Drift albo Concept Drift) ---
    retrain_flags = {}
    retrain_reasons = {}
    for model_type in MODEL_CLASSES:
        model_info = models_logs_by_name.get(model_type)

        if model_info is None:
            # Model tego typu jeszcze nigdy nie istniał w models_logs - nie ma
            # z czym porównywać driftu, więc wymuszamy pierwszy trening.
            retrain_flags[model_type] = True
            retrain_reasons[model_type] = "init"
            print(f"Brak wpisu dla '{model_type}' w models_logs - wymuszam pierwszy trening.")
            continue

        baseline_stats = model_info["baseline_stats"]
        selected_features = model_info["selected_features"]
        cd_stats = model_info["cd_stats"]

        dead_selected_features = [f for f in selected_features if f in dead_features]
        if dead_selected_features:
            retrain_flags[model_type] = True
            retrain_reasons[model_type] = f"DeadFeature({','.join(dead_selected_features)})"
            continue

        feature_drift_flags = compute_feature_drift_flags(
            today_data["returns"], baseline_stats, z_threshold=config["data_drift_z_threshold"]
        )
        data_drift, drifted_features = check_model_data_drift(
            feature_drift_flags, selected_features, pct_threshold=config["data_drift_pct_threshold"]
        )

        if data_drift:
            retrain_flags[model_type] = True
            retrain_reasons[model_type] = f"DD({','.join(drifted_features)})"
            # concept drift pomijany - retrening i tak wymuszony
        else:
            today_error = todays_errors.get(model_type)

            if today_error is None or naive_error is None:
                # brak dzisiejszej ewaluacji dla tego modelu albo dla naiwnego
                # punktu odniesienia (np. brak pending predykcji albo złoto
                # miało dziś święto) - nic nie sprawdzamy, stan detektora
                # zostaje bez zmian
                retrain_flags[model_type] = False
                continue

            concept_drift, new_cd_stats = update_page_hinkley(
                cd_stats, today_error - naive_error,
                delta=config["concept_drift_delta"], lambda_threshold=config["concept_drift_lambda"],
            )
            
            retrain_flags[model_type] = concept_drift
            retrain_reasons[model_type] = "CD" if concept_drift else None

            update_concept_drift_stats(model_type, model_info["id"], new_cd_stats)

    print(f"Wyniki drift detection: {retrain_flags}")

    # Predykcja dotyczy KOLEJNEGO dnia sesyjnego. BDay pomija tylko weekendy,
    # nie święta giełdowe - błąd wynikający ze świąt koryguje się samoczynnie,
    # dzień po dniu, przy kolejnych uruchomieniach.
    next_session_date = (as_of_date + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")

    # --- Krok 2: uruchomienie blackboxów (osobna pętla - retrain_flags jest
    # już w pełni policzone dla wszystkich modeli z pętli powyżej) ---
    todays_predictions = {}
    for model_type in MODEL_CLASSES:
        model_info = models_logs_by_name.get(model_type)
        storage_path = model_info["storage_path"] if model_info else None
        baseline_stats = model_info["baseline_stats"] if model_info else None

        model = MODEL_CLASSES[model_type](storage_path=storage_path, baseline_stats=baseline_stats, config=config)
        model.load_data(today_data)
        model.retrain_if_needed(retrain_flags[model_type], as_of_date)
        result = model.predict()

        if retrain_flags[model_type]:
            model_id = save_model_version_if_needed(
                model_type, result, model_info, retrain_reasons[model_type]
            )
        else:
            # retrain_flags[model_type] == False gwarantuje model_info != None
            # (bootstrap zawsze wymusza retrain=True) - bezpieczne bez .get().
            model_id = model_info["id"]

        llm_comment = build_llm_comment(model_type, result["shap_values"], result["predicted_value"])
        save_prediction(model_id, next_session_date, result, llm_comment)

        todays_predictions[model_type] = result
        print(f"Predykcja modelu '{model_type}': {result}")


if __name__ == "__main__":
    main()