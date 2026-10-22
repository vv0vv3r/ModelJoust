import sys
import traceback
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main_pipeline
from main_pipeline import main

DAYS_TO_SIMULATE = 100

# Komentarze LLM tylko dla prognoz z ostatniego symulowanego dnia (te widać
# w aplikacji na stronie "Dziś") - dla wcześniejszych dni pole zostaje puste.
# Setki wywołań Groq na raz wyczerpują limit API, a komentarze do
# historycznych prognoz nie są potrzebne.
LLM_ONLY_LAST_DAY = True

_build_llm_comment = main_pipeline.build_llm_comment
_llm_enabled = True


def _llm_comment_if_enabled(model_type, shap_values, predicted_value):
    if _llm_enabled:
        return _build_llm_comment(model_type, shap_values, predicted_value)
    return None


main_pipeline.build_llm_comment = _llm_comment_if_enabled


def run_simulation():
    """
    Uruchamia main() z main_pipeline.py kolejno dla DAYS_TO_SIMULATE
    ostatnich dni roboczych - test end-to-end całego pipeline'u na realnych
    danych historycznych. Błąd w jednym dniu nie przerywa symulacji,
    tylko jest odnotowywany i podsumowywany na końcu.

    Ostatni symulowany dzień = poprzedni dzień roboczy (wg czasu New York):
    dzisiejsza sesja może jeszcze trwać, a jej cenę zamknięcia i tak
    wieczorem zapisze dzienny przebieg z GitHub Actions.
    """
    global _llm_enabled
    end_date = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None) - pd.tseries.offsets.BDay(1)
    sim_dates = pd.bdate_range(end=end_date, periods=DAYS_TO_SIMULATE)

    print(f"Symulacja {len(sim_dates)} dni roboczych: {sim_dates[0].date()} -> {sim_dates[-1].date()}")

    failures = []
    for i, date in enumerate(sim_dates, start=1):
        _llm_enabled = not LLM_ONLY_LAST_DAY or i == len(sim_dates)
        print(f"\n{'=' * 80}\nDzień {i}/{len(sim_dates)}: {date.date()}\n{'=' * 80}")
        try:
            main(as_of_date=date)
        except Exception as e:
            print(f"BŁĄD w dniu {date.date()}: {e}")
            traceback.print_exc()
            failures.append((date.date(), str(e)))

    print(f"\n\nSymulacja zakończona. Błędów: {len(failures)}/{len(sim_dates)}.")
    for date, err in failures:
        print(f"  - {date}: {err}")


if __name__ == "__main__":
    run_simulation()
