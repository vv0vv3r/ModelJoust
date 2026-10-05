def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_param_grid(value, int_keys=(), float_keys=(), nullable_int_keys=()) -> str | None:
    if not isinstance(value, dict) or not value:
        return "Musi być niepustym słownikiem list."
    for hp, values in value.items():
        if not isinstance(values, list) or not values:
            return f"'{hp}': musi być niepustą listą wartości."
        for v in values:
            if hp in nullable_int_keys and v is None:
                continue
            if hp in int_keys and not (_is_int(v) and v > 0):
                return f"'{hp}': wartości muszą być dodatnimi liczbami całkowitymi."
            if hp in float_keys and not (_is_number(v) and 0 < v <= 1):
                return f"'{hp}': wartości muszą być liczbami w zakresie (0, 1]."
    return None


# Kategoria + zwięzłe wyjaśnienie efektu zmiany (w górę / w dół) + walidacja
# wprowadzonej wartości - używane wyłącznie do wyświetlenia w panelu admina
# (pages/6_admin.py); nie zmienia opisu zapisanego w wierszu pipeline_config.
CONFIG_META = {
    "data_drift_z_threshold": {
        "category": "Data Drift",
        "effect": (
            "Próg Z-score: o ile odchyleń standardowych od średniej z treningu musi odbiegać "
            "cecha, żeby uznać ją za zdryfowaną. Wyżej = mniej retreningów, ale ryzyko przegapienia "
            "realnej zmiany rynku. Niżej = czulszy, więcej fałszywych alarmów."
        ),
        "validate": lambda v: None if (_is_number(v) and v > 0) else "Musi być liczbą dodatnią.",
    },
    "data_drift_pct_threshold": {
        "category": "Data Drift",
        "effect": (
            "Jaki odsetek cech modelu musi zdryfować jednocześnie, żeby wymusić retrening. "
            "Wyżej = rzadsze, bardziej zachowawcze retreningi. Niżej = wystarczy dryf pojedynczej cechy."
        ),
        "validate": lambda v: None if (_is_number(v) and 0 <= v <= 1) else "Musi być w zakresie 0-1.",
    },
    "concept_drift_delta": {
        "category": "Concept Drift (Page-Hinkley)",
        "effect": (
            "Tolerancja odejmowana co dzień w liczniku Page-Hinkley, w jednostkach ułamka ceny "
            "(0.001 = 0,1 pp). Licznik śledzi błąd modelu MINUS błąd modelu naiwnego, więc "
            "delta to przewaga błędu nad naiwnym, którą jeszcze ignorujemy. Wyżej = dryf "
            "koncepcji wykrywany później. Niżej = szybsza reakcja, więcej fałszywych alarmów."
        ),
        "validate": lambda v: None if (_is_number(v) and v >= 0) else "Musi być liczbą nieujemną.",
    },
    "concept_drift_lambda": {
        "category": "Concept Drift (Page-Hinkley)",
        "effect": (
            "Próg, po którego przekroczeniu Page-Hinkley zgłasza dryf koncepcji, w jednostkach "
            "ułamka ceny (0.03 = 3 pp skumulowanej przewagi błędu modelu nad naiwnym). Alarm po "
            "ok. lambda / (pogorszenie - delta) sesjach. Wyżej = wolniejsza detekcja. Niżej = "
            "szybsza, ale więcej fałszywych alarmów."
        ),
        "validate": lambda v: None if (_is_number(v) and v > 0) else "Musi być liczbą dodatnią.",
    },
    "active_model_margin": {
        "category": "Champion / Challenger",
        "effect": (
            "O ile względnie (%) challenger musi pobić MAPE aktywnego modelu, żeby liczyć się jako "
            "wygrana danego dnia. Wyżej = trudniej przełączyć model, stabilniej. Niżej = łatwiej, "
            "ryzyko częstego przeskakiwania między modelami."
        ),
        "validate": lambda v: None if (_is_number(v) and 0 <= v < 1) else "Musi być w zakresie [0, 1).",
    },
    "active_model_streak_days": {
        "category": "Champion / Challenger",
        "effect": (
            "Ile dni z rzędu challenger musi wygrywać, żeby system go aktywował. Wyżej = bardziej "
            "zachowawcze, wolniejsze przełączanie. Niżej = szybsza reakcja, większe ryzyko "
            "przełączenia na podstawie krótkotrwałej fluktuacji."
        ),
        "validate": lambda v: None if (_is_int(v) and v >= 1) else "Musi być liczbą całkowitą >= 1.",
    },
    "rolling_mape_window_days": {
        "category": "Champion / Challenger",
        "effect": (
            "Z ilu ostatnich dni liczone jest kroczące MAPE porównujące modele. Wyżej = gładsze, "
            "mniej reaktywne. Niżej = bardziej zaszumione, ale szybciej odzwierciedla ostatnie zmiany."
        ),
        "validate": lambda v: None if (_is_int(v) and v >= 1) else "Musi być liczbą całkowitą >= 1.",
    },
    "training_window_years": {
        "category": "Trening — dane wejściowe",
        "effect": (
            "Ile ostatnich lat historii jest używane do treningu OLS/Lasso/RF/XGBoost (nie cała "
            "dostępna historia od początku). Wyżej = więcej danych, stabilniejsza selekcja cech, "
            "ale ryzyko mieszania starych reżimów rynkowych (inne stopy procentowe, inny reżim QE). "
            "Niżej = bardziej aktualne, ale mniej danych - mniej stabilna selekcja i wyższe VIF/p-value."
        ),
        "validate": lambda v: None if (_is_int(v) and v >= 1) else "Musi być liczbą całkowitą >= 1 (lata).",
    },
    "ols_p_value_threshold": {
        "category": "Trening — OLS",
        "effect": (
            "Próg istotności (p-value) przy eliminacji wstecznej cech. Wyżej = luźniejsze kryterium, "
            "model zostawia więcej (słabszych) cech. Niżej = ostrzejsze, tylko mocno istotne cechy."
        ),
        "validate": lambda v: None if (_is_number(v) and 0 < v <= 1) else "Musi być w zakresie (0, 1].",
    },
    "ols_vif_threshold": {
        "category": "Trening — OLS",
        "effect": (
            "Próg VIF przy filtrowaniu współliniowości cech. Wyżej = toleruje więcej skorelowanych "
            "cech, mniej stabilne współczynniki. Niżej = ostrzejszy filtr, mniej, ale bardziej "
            "niezależnych cech."
        ),
        "validate": lambda v: None if (_is_number(v) and v > 1) else "Musi być liczbą większą niż 1 (teoretyczne minimum VIF).",
    },
    "tree_cumulative_importance_threshold": {
        "category": "Trening — Random Forest / XGBoost",
        "effect": (
            "Jaki odsetek skumulowanej ważności cech (feature importance) ma pokryć wybrany zestaw. "
            "Wyżej = więcej cech potrzebnych, bardziej złożony model. Niżej = tylko kilka dominujących cech."
        ),
        "validate": lambda v: None if (_is_number(v) and 0 < v <= 1) else "Musi być w zakresie (0, 1].",
    },
    "cv_splits": {
        "category": "Trening — walidacja krzyżowa",
        "effect": (
            "Liczba foldów walidacji krzyżowej przy doborze hiperparametrów (RF, XGBoost, Lasso) - "
            "wszystkie trzy modele używają chronologicznego TimeSeriesSplit."
            "Wyżej = solidniejszy dobór, wolniejszy trening. Niżej = szybciej, "
            "ale bardziej zaszumiony wybór."
        ),
        "validate": lambda v: None if (_is_int(v) and v >= 2) else "Musi być liczbą całkowitą >= 2.",
    },
    "rf_param_grid": {
        "category": "Trening — Random Forest",
        "effect": (
            "Siatka GridSearchCV. Więcej wartości/hiperparametrów = wolniejsze, dokładniejsze "
            "strojenie. n_estimators/min_samples_leaf: dodatnie liczby całkowite, max_depth: "
            "dodatnia liczba całkowita lub null."
        ),
        "validate": lambda v: _validate_param_grid(
            v, int_keys={"n_estimators", "min_samples_leaf"}, nullable_int_keys={"max_depth"}
        ),
    },
    "xgboost_param_grid": {
        "category": "Trening — XGBoost",
        "effect": (
            "Siatka GridSearchCV. Więcej wartości/hiperparametrów = wolniejsze, dokładniejsze "
            "strojenie. n_estimators/max_depth: dodatnie liczby całkowite, learning_rate: liczba "
            "w zakresie (0, 1]."
        ),
        "validate": lambda v: _validate_param_grid(
            v, int_keys={"n_estimators", "max_depth"}, float_keys={"learning_rate"}
        ),
    },
}
