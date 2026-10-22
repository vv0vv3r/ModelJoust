import sys
import traceback
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_pipeline import main

DAYS_TO_SIMULATE = 100


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
    end_date = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None) - pd.tseries.offsets.BDay(1)
    sim_dates = pd.bdate_range(end=end_date, periods=DAYS_TO_SIMULATE)

    print(f"Symulacja {len(sim_dates)} dni roboczych: {sim_dates[0].date()} -> {sim_dates[-1].date()}")

    failures = []
    for i, date in enumerate(sim_dates, start=1):
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
