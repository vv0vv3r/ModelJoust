"""
Atrapa klienta Supabase trzymająca wszystkie tabele w pamięci - pozwala
uruchomić main_pipeline.main() lokalnie, bez połączenia z bazą i bez
żadnej zmiany w kodzie systemu (run_simulation.py podstawia ją pod
db_client.supabase przed importem pipeline'u).

Obsługuje dokładnie ten podzbiór API supabase-py, którego używa system:
table().select/insert/update/upsert/delete, filtry eq/neq/gt/gte/lt/lte/is_
(także z .not_), order, limit, range, execute(), oraz storage.from_()
.upload/download. Odwzorowuje też zachowanie bazy, na którym kod polega:
kolumny autonumerowane (GENERATED ALWAYS AS IDENTITY), wartości domyślne
kolumn i ograniczenia unikalności z database/database_setup.sql.

Każdy odczyt i zapis kopiuje dane (deepcopy) - tak jak prawdziwa baza, kod
dostaje własną kopię i jej modyfikacja nie zmienia zawartości tabeli.
"""
import copy
import json
from pathlib import Path

import pandas as pd

# Kolumny i ich wartości domyślne wg database/database_setup.sql (created_at
# pominięte - kod systemu go nie czyta).
SCHEMA = {
    "raw_data": {
        "columns": {"target_date": None, "features": None, "actual_y": None},
        "identity": None,
        "unique": [("target_date",)],
    },
    "models_logs": {
        "columns": {
            "id": None, "model_type": None, "model_version": None, "is_active": False,
            "retrain_trigger": None, "train_start_date": None, "train_end_date": None,
            "selected_features": None, "baseline_stats": None, "storage_path": None,
            "concept_drift_stats": None,
        },
        "identity": "id",
        "unique": [("id",)],
    },
    "model_predictions": {
        "columns": {
            "prediction_id": None, "target_date": None, "model_id": None,
            "predicted_value": None, "actual_value": None, "error_value": None,
            "shap_values": None, "llm_comment": None, "status": "pending",
        },
        "identity": "prediction_id",
        "unique": [("prediction_id",), ("target_date", "model_id")],
    },
    "system_logs": {
        "columns": {"log_date": None, "active_model": None, "mape": None, "beats_active": None},
        "identity": None,
        "unique": [("log_date",)],
    },
    "pipeline_config": {
        "columns": {"id": None, "key": None, "value": None, "description": None, "active": True},
        "identity": "id",
        "unique": [("id",)],
    },
}

# Tabele zapisywane do CSV po symulacji (raw_data to dane wejściowe, nie wynik).
RESULT_TABLES = ["models_logs", "model_predictions", "system_logs", "pipeline_config"]


class FakeResponse:
    def __init__(self, data):
        self.data = data


def _sort_key(value):
    # NULL "największy" - jak w Postgresie: na końcu przy ASC, na początku
    # przy DESC (sort z reverse=True).
    return (value is None, value)


class FakeQuery:
    def __init__(self, db: "FakeSupabase", table: str):
        self._db = db
        self._table = table
        self._action = "select"
        self._columns = None
        self._payload = None
        self._on_conflict = None
        self._filters = []
        self._negate_next = False
        self._orders = []
        self._limit = None
        self._range = None

    # --- rodzaj operacji ---
    def select(self, columns: str = "*"):
        self._action = "select"
        self._columns = None if columns.strip() == "*" else [c.strip() for c in columns.split(",")]
        return self

    def insert(self, payload):
        self._action, self._payload = "insert", payload
        return self

    def update(self, payload: dict):
        self._action, self._payload = "update", payload
        return self

    def upsert(self, payload, on_conflict: str | None = None):
        self._action, self._payload = "upsert", payload
        self._on_conflict = [c.strip() for c in on_conflict.split(",")] if on_conflict else None
        return self

    def delete(self):
        self._action = "delete"
        return self

    # --- filtry ---
    @property
    def not_(self):
        self._negate_next = True
        return self

    def _add_filter(self, column, check):
        negate, self._negate_next = self._negate_next, False
        self._filters.append((column, check, negate))
        return self

    def eq(self, column, value):
        return self._add_filter(column, lambda v: v is not None and v == value)

    def neq(self, column, value):
        return self._add_filter(column, lambda v: v is not None and v != value)

    def gt(self, column, value):
        return self._add_filter(column, lambda v: v is not None and v > value)

    def gte(self, column, value):
        return self._add_filter(column, lambda v: v is not None and v >= value)

    def lt(self, column, value):
        return self._add_filter(column, lambda v: v is not None and v < value)

    def lte(self, column, value):
        return self._add_filter(column, lambda v: v is not None and v <= value)

    def is_(self, column, value):
        if value not in ("null", None):
            raise NotImplementedError(f"is_() obsługuje tylko 'null', dostał {value!r}")
        return self._add_filter(column, lambda v: v is None)

    # --- sortowanie i stronicowanie ---
    def order(self, column, desc: bool = False):
        self._orders.append((column, desc))
        return self

    def limit(self, n: int):
        self._limit = n
        return self

    def range(self, start: int, end: int):
        self._range = (start, end)
        return self

    # --- wykonanie ---
    def _matches(self, row: dict) -> bool:
        for column, check, negate in self._filters:
            if check(row.get(column)) == negate:
                return False
        return True

    def execute(self) -> FakeResponse:
        rows = self._db._tables[self._table]
        if self._action == "select":
            return FakeResponse(self._run_select(rows))
        if self._action == "insert":
            return FakeResponse(self._db._insert(self._table, self._payload))
        if self._action == "update":
            return FakeResponse(self._run_update(rows))
        if self._action == "upsert":
            return FakeResponse(self._db._upsert(self._table, self._payload, self._on_conflict))
        if self._action == "delete":
            kept, removed = [], []
            for row in rows:
                (removed if self._matches(row) else kept).append(row)
            self._db._tables[self._table] = kept
            return FakeResponse(copy.deepcopy(removed))
        raise NotImplementedError(self._action)

    def _run_select(self, rows: list[dict]) -> list[dict]:
        result = [row for row in rows if self._matches(row)]
        # Sortowanie od ostatniego klucza do pierwszego (stabilny sort) daje
        # kolejność jak ORDER BY a, b.
        for column, desc in reversed(self._orders):
            result.sort(key=lambda r: _sort_key(r.get(column)), reverse=desc)
        if self._range is not None:
            start, end = self._range
            result = result[start:end + 1]
        if self._limit is not None:
            result = result[:self._limit]
        if self._columns is not None:
            result = [{c: row.get(c) for c in self._columns} for row in result]
        return copy.deepcopy(result)

    def _run_update(self, rows: list[dict]) -> list[dict]:
        if not self._filters:
            raise ValueError("update() bez filtra - prawdziwa baza też by tego nie przepuściła.")
        updated = []
        for row in rows:
            if self._matches(row):
                candidate = {**row, **copy.deepcopy(self._payload)}
                self._db._check_unique(self._table, candidate, ignore=row)
                row.update(copy.deepcopy(self._payload))
                updated.append(copy.deepcopy(row))
        return updated


class FakeBucket:
    def __init__(self, files: dict):
        self._files = files

    def upload(self, path: str, file: bytes, file_options: dict | None = None):
        upsert = str((file_options or {}).get("upsert", "false")).lower() == "true"
        if path in self._files and not upsert:
            raise ValueError(f"Plik '{path}' już istnieje w storage.")
        self._files[path] = bytes(file)
        return {"path": path}

    def download(self, path: str) -> bytes:
        if path not in self._files:
            raise ValueError(f"Brak pliku '{path}' w storage.")
        return self._files[path]


class FakeStorage:
    def __init__(self):
        self._buckets: dict[str, dict] = {}

    def from_(self, bucket: str) -> FakeBucket:
        return FakeBucket(self._buckets.setdefault(bucket, {}))


class FakeSupabase:
    def __init__(self):
        self._tables = {name: [] for name in SCHEMA}
        self._next_id = {name: 1 for name in SCHEMA}
        self.storage = FakeStorage()

    def table(self, name: str) -> FakeQuery:
        if name not in SCHEMA:
            raise ValueError(f"Nieznana tabela '{name}'.")
        return FakeQuery(self, name)

    # --- zapis z odwzorowaniem zachowania bazy ---
    def _new_row(self, table: str, payload: dict) -> dict:
        schema = SCHEMA[table]
        unknown = set(payload) - set(schema["columns"])
        if unknown:
            raise ValueError(f"Nieznane kolumny w '{table}': {unknown}")
        identity = schema["identity"]
        if identity and identity in payload:
            raise ValueError(f"Kolumna '{identity}' jest GENERATED ALWAYS - nie wolno jej podawać.")
        row = {**copy.deepcopy(schema["columns"]), **copy.deepcopy(payload)}
        if identity:
            row[identity] = self._next_id[table]
            self._next_id[table] += 1
        return row

    def _check_unique(self, table: str, row: dict, ignore: dict | None = None) -> None:
        for columns in SCHEMA[table]["unique"]:
            key = tuple(row.get(c) for c in columns)
            for existing in self._tables[table]:
                if existing is ignore:
                    continue
                if tuple(existing.get(c) for c in columns) == key:
                    raise ValueError(f"Naruszenie unikalności {columns} w '{table}': {key}")
        if table == "pipeline_config" and row.get("active"):
            for existing in self._tables[table]:
                if existing is not ignore and existing["active"] and existing["key"] == row["key"]:
                    raise ValueError(f"Drugi aktywny wiersz pipeline_config dla '{row['key']}'.")

    def _insert(self, table: str, payload) -> list[dict]:
        payloads = payload if isinstance(payload, list) else [payload]
        new_rows = []
        for item in payloads:
            row = self._new_row(table, item)
            self._check_unique(table, row)
            self._tables[table].append(row)
            new_rows.append(row)
        return copy.deepcopy(new_rows)

    def _upsert(self, table: str, payload, on_conflict: list[str] | None) -> list[dict]:
        if not on_conflict:
            raise ValueError("upsert() bez on_conflict nie jest używany przez system.")
        payloads = payload if isinstance(payload, list) else [payload]
        result = []
        for item in payloads:
            key = tuple(item.get(c) for c in on_conflict)
            existing = next(
                (r for r in self._tables[table] if tuple(r.get(c) for c in on_conflict) == key), None
            )
            if existing is None:
                result.extend(self._insert(table, item))
            else:
                existing.update(copy.deepcopy(item))
                result.append(copy.deepcopy(existing))
        return result

    # --- wejście/wyjście symulacji ---
    def reset_results(self) -> None:
        """Czysta baza przed kolejną symulacją: wszystko poza raw_data
        (dane wejściowe zostają) - tabele, liczniki id i pliki modeli."""
        for table in SCHEMA:
            if table != "raw_data":
                self._tables[table] = []
                self._next_id[table] = 1
        self.storage = FakeStorage()

    def load_raw_data(self, market_df: pd.DataFrame) -> None:
        """Wgrywa dane rynkowe (indeks = data, kolumny Y_Gold + X_...) do
        raw_data w tym samym formacie co seed_historical_data.py: braki jako
        null, bez ffill."""
        rows = []
        for date_index, row in market_df.iterrows():
            values = {k: (None if pd.isna(v) else float(v)) for k, v in row.to_dict().items()}
            actual_y = values.pop("Y_Gold", None)
            rows.append({
                "target_date": pd.Timestamp(date_index).strftime("%Y-%m-%d"),
                "features": values,
                "actual_y": actual_y,
            })
        self._tables["raw_data"] = []
        self._insert("raw_data", rows)

    def dump_tables(self, folder: Path) -> None:
        """Zapisuje tabele wynikowe do CSV (kolumny JSON jako tekst JSON)."""
        folder.mkdir(parents=True, exist_ok=True)
        for table in RESULT_TABLES:
            rows = [
                {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
                for row in self._tables[table]
            ]
            pd.DataFrame(rows, columns=list(SCHEMA[table]["columns"])).to_csv(
                folder / f"{table}.csv", index=False
            )
