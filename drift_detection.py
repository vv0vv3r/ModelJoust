import pandas as pd

from db_client import supabase




def fetch_recent_window(as_of_date: pd.Timestamp, days_back: int = 15) -> pd.DataFrame:
    """Pobiera okno z raw_data, potrzebne do zbudowania kompletnego,
    ffillowanego wiersza na dziś (bez nulli). as_of_date przyjmowane z
    zewnątrz (nie liczone tu ponownie), żeby cały pipeline operował na
    jednej, spójnej dacie "dziś"."""
    cutoff_date = (as_of_date - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    as_of_date_str = as_of_date.strftime("%Y-%m-%d")

    # lte(as_of_date) jest KRYTYCZNE przy symulacji/nadrabianiu zaległych dni -
    # bez tego okno łapałoby też wiersze z przyszłości (względem as_of_date),
    # które już mogą istnieć w raw_data (np. po jednorazowym zaseedowaniu
    # wielu lat historii naraz). Przy codziennym, żywym użyciu raw_data i tak
    # nigdy nie ma wierszy nowszych niż dziś, więc to nie zmienia zachowania.
    response = (
        supabase.table("raw_data")
        .select("target_date, features, actual_y")
        .gte("target_date", cutoff_date)
        .lte("target_date", as_of_date_str)
        .order("target_date")
        .execute()
    )

    rows = response.data
    if not rows:
        raise ValueError("Brak jakichkolwiek danych w raw_data dla żądanego okna.")

    df = pd.DataFrame(rows)
    df["target_date"] = pd.to_datetime(df["target_date"])
    features_df = pd.json_normalize(df["features"])
    features_df.index = df["target_date"]
    features_df["actual_y"] = df["actual_y"].values

    return features_df.sort_index()


def fetch_full_history_returns(as_of_date: pd.Timestamp, window_years: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """
    Pobiera historię raw_data z ostatnich `window_years` lat DO as_of_date
    włącznie (NIE całą dostępną historię od początku) i zwraca (returns_df,
    dead_features): stopy zwrotu dzień-do-dnia dla wszystkich kolumn (cechy +
    actual_y), liczone na ffillowanych poziomach, oraz listę martwych cech.
    Współdzielona przez wszystkie blackboxy statystyczne/ML przy retreningu.

    Okno jest CELOWO ograniczone, nie "cała historia od początku" - zależności
    między złotem a innymi instrumentami nie są stacjonarne w wieloletniej
    skali (różne reżimy stóp procentowych/QE), więc zbyt długie okno
    rozcieńczałoby aktualny reżim rynkowy starymi obserwacjami i osłabiało
    sens mechanizmu drift-detection (retrening ma reagować na ZMIANĘ reżimu,
    nie być zdominowany przez uśrednioną historię sprzed lat). Efekt uboczny:
    baseline_stats (Data Drift) też liczą się już tylko z tego okna, nie
    całej historii - spójne z tym, na czym model faktycznie się trenuje.

    dead_features MUSI być liczone na SUROWYCH (nieffillowanych) poziomach,
    przed przekazaniem do compute_dead_features - ffill zamienia permanentnie
    martwą kolumnę w stałą wartość, a pct_change() stałej to 0.0, nie NaN, więc
    na już przetworzonych zwrotach compute_dead_features nigdy by niczego nie
    wykrył (patrz compute_dead_features).

    Paginacja przez .range() jest KRYTYCZNA - Supabase domyślnie zwraca
    maksymalnie 1000 wierszy na zapytanie. Bez tego trening cichutko
    obcinałby się do najstarszych 1000 wierszy i ignorował wszystko nowsze,
    bez żadnego błędu/ostrzeżenia.

    lte(as_of_date) jest KRYTYCZNE przy symulacji/nadrabianiu zaległych dni -
    bez tego trening "widziałby" dane z przyszłości względem as_of_date
    (lookahead bias).
    """
    as_of_date_str = as_of_date.strftime("%Y-%m-%d")
    cutoff_date_str = (as_of_date - pd.DateOffset(years=window_years)).strftime("%Y-%m-%d")
    page_size = 1000
    all_rows = []
    start = 0
    while True:
        response = (
            supabase.table("raw_data")
            .select("target_date, features, actual_y")
            .gte("target_date", cutoff_date_str)
            .lte("target_date", as_of_date_str)
            .order("target_date")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = response.data
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size

    if not all_rows:
        raise ValueError("Brak jakichkolwiek danych w raw_data do treningu.")

    df = pd.DataFrame(all_rows)
    df["target_date"] = pd.to_datetime(df["target_date"])
    features_df = pd.json_normalize(df["features"])
    features_df.index = df["target_date"]
    features_df["actual_y"] = df["actual_y"].values
    features_df = features_df.sort_index()

    dead_features = compute_dead_features(features_df)

    filled = features_df.ffill()
    return filled.pct_change(), dead_features


def build_today_return_row(window_df: pd.DataFrame) -> dict:
    """
    Zwraca dzisiejsze dane wejściowe dla drift checku i predict() każdego
    blackboxa - liczone tylko raz na dzień, reużywane niżej.

    Ceny (poziomy) są niestacjonarne (random walk) - Z-score liczony na
    surowych poziomach albo regresja z opóźnionym poziomem jako predyktorem
    "driftowałaby"/oszukiwałaby się niemal zawsze, niezależnie od realnej
    zmiany reżimu rynkowego. Dlatego zwracamy DWIE rzeczy o różnym przeznaczeniu:
      - "returns": dzienna stopa zwrotu (dziś vs wczoraj) KAŻDEJ cechy,
        łącznie z actual_y (potencjalny człon autoregresyjny AR(1)) - to
        wejście do drift checku i do modeli statystycznych/ML.
      - "last_actual_y_level": ostatnia znana cena złota jako poziom (nie
        zwrot) - potrzebna modelowi naiwnemu oraz do konwersji
        "przewidziany zwrot -> przewidziana cena" w pozostałych modelach.
    """
    filled = window_df.ffill()
    returns_row = filled.pct_change().iloc[-1]

    return {
        "returns": returns_row.to_dict(),
        "last_actual_y_level": float(filled["actual_y"].iloc[-1]),
    }

def compute_dead_features(df: pd.DataFrame, lookback: int = 10) -> list[str]:
    """
    Zwraca cechy, dla których w ostatnich `lookback` wierszach NIE MA ani
    jednej wartości - sygnał, że dostawca danych (yfinance) przestał w ogóle
    raportować ten instrument, a nie zwykła przerwa świąteczna (te ffill
    wypełnia bez problemu - patrz build_today_return_row/
    fetch_full_history_returns). Binarny check (nie procentowy) - nawet
    najdłuższa realna przerwa giełdowa (japoński Nowy Rok, ~4 sesje) jest
    wyraźnie krótsza niż `lookback`, więc zostaje margines na przejściowe
    problemy API.
    """
    recent = df.tail(lookback)
    return [col for col in df.columns if recent[col].isna().all()]


def fetch_active_models_info() -> dict:
    """
    Pobiera z models_logs informacje o aktualnie aktywnych (is_active=True)
    wersjach każdego z 3 modeli: baseline_stats, selected_features (do Data
    Drift), cd_stats (stan detektora Page-Hinkley, do Concept Drift), id i
    storage_path (do wczytania/zapisania obiektu modelu w blackboxie).
    """
    response = (
        supabase.table("models_logs")
        .select("id, model_type, baseline_stats, selected_features, concept_drift_stats, storage_path")
        .eq("is_active", True)
        .execute()
    )

    rows = response.data
    if not rows:
        # Legalny stan startowy (pierwsze uruchomienie systemu, jeszcze zero
        # modeli) - main_pipeline.py obsługuje brak wpisu per model_type
        # przez bootstrap (wymuszony pierwszy trening), nie błąd.
        print("Brak jakichkolwiek aktywnych wpisów w models_logs - zakładam pierwsze uruchomienie systemu.")
        return {}

    models_info = {}
    for row in rows:
        models_info[row["model_type"]] = {
            "id": row["id"],
            "baseline_stats": row["baseline_stats"],
            "selected_features": row["selected_features"],
            "cd_stats": row["concept_drift_stats"],
            "storage_path": row["storage_path"],
        }

    return models_info

def compute_feature_drift_flags(today_returns: dict, baseline_stats: dict,
                                  z_threshold: float = 3.0) -> dict:
    """
    Liczy z-score dla KAŻDEJ cechy, dla której model ma baseline_stats
    (czyli wszystkich cech z jego okna treningowego, nie tylko selected_features).
    Zwraca {cecha: True/False} - czy ta konkretna cecha wykazuje dziś drift.

    today_returns i baseline_stats muszą być w tych samych jednostkach -
    stopy zwrotu, nie poziomy cen (patrz build_today_return_row). Inaczej
    porównanie nie ma sensu (niestacjonarność poziomów -> pozorny,
    permanentny drift).
    """
    flags = {}
    for feature, stats in (baseline_stats or {}).items():
        today_value = today_returns.get(feature)
        if today_value is None or pd.isna(today_value):
            continue

        mean = stats["mean"]
        std = stats["std"]
        if std == 0:
            continue

        z_score = (today_value - mean) / std
        flags[feature] = abs(z_score) > z_threshold

    return flags


def check_model_data_drift(feature_drift_flags: dict, selected_features: list,
                             pct_threshold: float = 0.25) -> tuple[bool, list[str]]:
    """
    Filtruje pełny słownik flag drift do TYLKO selected_features danego modelu,
    sprawdza czy odsetek cech z driftem przekracza próg procentowy, i zwraca
    też listę nazw zdryfowanych cech (do zapisania w retrain_trigger).

    UWAGA: próg jest PROCENTOWY, nie bezwzględny - modele z mniejszą liczbą
    selected_features są przez to nieproporcjonalnie bardziej wrażliwe na
    drift.
    """
    relevant_features = [f for f in selected_features if f in feature_drift_flags]

    if not relevant_features:
        return False, []

    drifted_features = [f for f in relevant_features if feature_drift_flags[f]]
    drifted_ratio = len(drifted_features) / len(relevant_features)
    return drifted_ratio >= pct_threshold, drifted_features

def update_page_hinkley(cd_stats: dict | None, today_error: float,
                          delta: float = 0.005, lambda_threshold: float = 5.0) -> tuple[bool, dict]:
    """
    Aktualizuje stan detektora Page-Hinkley o dzisiejszą obserwację.
    today_error: błąd modelu MINUS błąd modelu naiwnego z tego samego dnia,
    oba jako ułamek ceny (0.01 = 1 pp) - patrz main_pipeline.py. delta i
    lambda_threshold muszą być w tych samych jednostkach (ułamek, nie %).
    Alarm po ok. lambda / (wzrost_sygnału - delta) sesjach utrzymującego się
    pogorszenia względem modelu naiwnego.
    cd_stats: {"mean_error":.., "n":.., "cumulative_sum":.., "min_cumulative_sum":..} albo None.
    Zwraca (drift_detected, nowy_cd_stats).
    """
    if not cd_stats:
        mean_error = 0.0
        n = 0
        cumulative_sum = 0.0
        min_cumulative_sum = 0.0
    else:
        mean_error = cd_stats.get("mean_error", 0.0)
        n = cd_stats.get("n", 0)
        cumulative_sum = cd_stats.get("cumulative_sum", 0.0)
        min_cumulative_sum = cd_stats.get("min_cumulative_sum", 0.0)

    n += 1
    mean_error += (today_error - mean_error) / n
    cumulative_sum += today_error - mean_error - delta
    min_cumulative_sum = min(min_cumulative_sum, cumulative_sum)

    drift_detected = (cumulative_sum - min_cumulative_sum) > lambda_threshold

    if drift_detected:
        # po wykryciu (i zaplanowanym retreningu) zaczynamy śledzić od nowa
        mean_error = 0.0
        n = 0
        cumulative_sum = 0.0
        min_cumulative_sum = 0.0

    new_cd_stats = {
        "mean_error": mean_error,
        "n": n,
        "cumulative_sum": cumulative_sum,
        "min_cumulative_sum": min_cumulative_sum,
    }
    return drift_detected, new_cd_stats