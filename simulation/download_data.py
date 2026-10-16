"""
Jednorazowe pobranie z yfinance historii wszystkich instrumentów systemu
do simulation/data/market_data.csv - symulacje czytają dane z tego pliku
zamiast odpytywać API w każdym dniu.

Ta sama lista tickerów i to samo czyszczenie (normalize_market_data) co
w systemie. Zapis bez ffill - braki zostają puste, jak w raw_data.

Uruchomienie (z katalogu repo): python -m simulation.download_data
"""
from pathlib import Path

import simulation.sandbox  # noqa: F401  (atrapy przed importem kodu systemu)

import pandas as pd
import yfinance as yf

from data_download import tickers, normalize_market_data

# Zapas na najdłuższe okno treningowe (5 lat) przed najwcześniejszym
# okresem symulacji. Ethereum jest w yfinance od 11.2017.
START_DATE = "2018-01-01"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "market_data.csv"


def download_market_data() -> pd.DataFrame:
    end_date = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # end jest wyłączny
    df_raw = yf.download(
        list(tickers.values()), start=START_DATE, end=end_date, interval="1d", progress=False
    )["Close"]
    df = normalize_market_data(df_raw)
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    df.index.name = "target_date"
    return df[list(tickers.keys())]


if __name__ == "__main__":
    df = download_market_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH)

    print(f"Zapisano {len(df)} dni ({df.index[0]} -> {df.index[-1]}) do {OUTPUT_PATH}")
    first_valid = df.apply(lambda col: col.first_valid_index())
    print("Instrumenty z najpóźniejszym początkiem notowań:")
    print(first_valid.sort_values(ascending=False).head(5).to_string())
