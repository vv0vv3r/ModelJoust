"""
Analiza wrażliwości: symulacja każdej kombinacji z PARAM_GRID
(settings.py) na okresie strojenia, z rozbiegiem.

Bezpieczna do przerwania i wznowienia: kombinacja, której folder ma już
meta.json (zapisywany na samym końcu symulacji), jest pomijana. Po każdej
kombinacji simulation/results/podsumowanie.csv jest budowane od nowa ze
wszystkich zakończonych symulacji.

Uruchomienie (z katalogu repo): python -m simulation.run_grid
"""
import itertools

from simulation.run_simulation import RESULTS_DIR, run_simulation
from simulation.settings import PARAM_GRID, TUNING_END, TUNING_START, folder_name, simulation_start
from simulation.summarize import build_grid_summary

SUMMARY_PATH = RESULTS_DIR / "podsumowanie.csv"


def run_grid() -> None:
    keys = list(PARAM_GRID)
    combos = [dict(zip(keys, values)) for values in itertools.product(*PARAM_GRID.values())]
    start = simulation_start(TUNING_START)
    print(f"Siatka: {len(combos)} kombinacji, symulacja {start} -> {TUNING_END} "
          f"(oceniane od {TUNING_START}).")

    for i, params in enumerate(combos, start=1):
        name = folder_name(params)
        if (RESULTS_DIR / name / "meta.json").exists():
            print(f"[{i}/{len(combos)}] {name} - już zrobione, pomijam.")
            continue

        print(f"\n[{i}/{len(combos)}] {name}")
        run_simulation(name, start, TUNING_END, params)

        build_grid_summary(RESULTS_DIR).to_csv(SUMMARY_PATH, index=False)

    print(f"\nWszystkie kombinacje zrobione. Podsumowanie: {SUMMARY_PATH}")


if __name__ == "__main__":
    run_grid()
