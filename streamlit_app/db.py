import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError(
        "Brak SUPABASE_URL / SUPABASE_ANON_KEY w pliku .env! "
        "To osobny klucz (anon, publiczny) od tego używanego przez pipeline "
        "(service_role) - patrz Project Settings -> API w Supabase."
    )

# Klucz anon, ograniczony przez RLS do samego odczytu - apka nigdy nie pisze
# do bazy tym klientem. Zapis (panel admina) dostaje osobnego klienta na
# service_role, tworzonego dopiero po weryfikacji hasła.
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Supabase zwraca najwyżej tyle wierszy na jedno zapytanie.
PAGE_SIZE = 1000


def fetch_all(build_query, max_rows: int | None = None) -> list[dict]:
    """Wszystkie wiersze zapytania (albo pierwsze max_rows), pobierane
    stronami po PAGE_SIZE - bez tego zapytanie o dłuższą historię cicho
    obcinałoby wynik do 1000 wierszy. build_query: funkcja zwracająca NOWE
    zapytanie z filtrami i jednoznacznym sortowaniem (np. dodatkowo po
    kluczu), żeby kolejne strony się nie nakładały."""
    rows = []
    start = 0
    while max_rows is None or len(rows) < max_rows:
        end = start + PAGE_SIZE - 1
        if max_rows is not None:
            end = min(end, max_rows - 1)
        batch = build_query().range(start, end).execute().data or []
        rows.extend(batch)
        if len(batch) < end - start + 1:
            break
        start = end + 1
    return rows


def last_n_evaluated_cutoff(n: int) -> tuple[str | None, int]:
    # "N dni" w całej apce = N dni z FAKTYCZNIE ewaluowaną predykcją, nie N
    # ostatnich wierszy system_logs - dzień bez notowania giełdowego (status
    # 'unused') albo dzień, w którym pipeline w ogóle się nie uruchomił, nie
    # powinien "zjadać" miejsca w oknie kosztem realnych danych. Limit wiersza
    # to n*5+20 (max 5 modeli/dzień + zapas), żeby złapać z pewnością co
    # najmniej n różnych dat (stronami - dla 365 dni to więcej niż 1000
    # wierszy). Zwraca (cutoff, ile dat realnie znaleziono) - drugi element
    # pozwala wywołującemu wykryć sytuację "mamy mniej historii niż żądane N".
    rows = fetch_all(
        lambda: supabase.table("model_predictions")
        .select("target_date")
        .eq("status", "evaluated")
        .order("target_date", desc=True)
        .order("prediction_id"),
        max_rows=n * 5 + 20,
    )
    dates = sorted({row["target_date"] for row in rows}, reverse=True)
    if not dates:
        return None, 0
    window = dates[:n]
    return window[-1], len(window)
