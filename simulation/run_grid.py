"""
Analiza wrażliwości: symulacja każdej kombinacji z PARAM_GRID
(settings.py) na okresie strojenia, z rozbiegiem.

Bezpieczna do przerwania i wznowienia: kombinacja, której folder zawiera
zakończoną symulację zgodną z bieżącymi ustawieniami (grid_run_problem w
summarize.py), jest pomijana. Folder z innych ustawień (inny okres albo
konfiguracja) jest liczony od nowa. Po każdej kombinacji
simulation/results/podsumowanie.csv jest budowane od nowa ze wszystkich
zakończonych symulacji.

Uruchomienie (z katalogu repo): python -m simulation.run_grid
"""
import itertools
import json

from simulation.run_simulation import RESULTS_DIR, run_simulation
from simulation.settings import PARAM_GRID, TUNING_END, TUNING_START, folder_name, simulation_start
from simulation.summarize import build_grid_summary, grid_run_problem

SUMMARY_PATH = RESULTS_DIR / "podsumowanie.csv"


def run_grid() -> None:
    keys = list(PARAM_GRID)
    combos = [dict(zip(keys, values)) for values in itertools.product(*PARAM_GRID.values())]
    names = [folder_name(params) for params in combos]
    if len(set(names)) != len(names):
        raise ValueError("Dwie kombinacje dają tę samą nazwę folderu - popraw folder_name w settings.py.")

    start = simulation_start(TUNING_START)
    print(f"Siatka: {len(combos)} kombinacji, symulacja {start} -> {TUNING_END} "
          f"(oceniane od {TUNING_START}).")

    for i, (params, name) in enumerate(zip(combos, names), start=1):
        folder = RESULTS_DIR / name
        problem = grid_run_problem(folder) if folder.exists() else "brak folderu"
        if problem is None:
            print(f"[{i}/{len(combos)}] {name} - już zrobione, pomijam.")
            continue
        if folder.exists():
            print(f"[{i}/{len(combos)}] {name} - liczę od nowa: {problem}.")

        print(f"\n[{i}/{len(combos)}] {name}")
        try:
            out_dir = run_simulation(name, start, TUNING_END, params)
        except Exception as e:
            # Pętla idzie dalej; folder bez meta.json zostanie policzony od
            # nowa przy kolejnym uruchomieniu. Ctrl+C (KeyboardInterrupt) nie
            # jest Exception, więc nadal zatrzymuje siatkę.
            print(f"BŁĄD symulacji {name}: {e!r} - pomijam, do powtórzenia przy kolejnym uruchomieniu.")
            continue

        failures = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))["failures"]
        if failures:
            print(f"UWAGA: {len(failures)} dni z błędem w {name} - szczegóły w meta.json i pipeline.log.")

        build_grid_summary(RESULTS_DIR).to_csv(SUMMARY_PATH, index=False)

    print(f"\nWszystkie kombinacje zrobione. Podsumowanie: {SUMMARY_PATH}")


if __name__ == "__main__":
    run_grid()
