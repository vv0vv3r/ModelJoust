"""
Podstawia atrapy pod moduły systemu, które łączą się z usługami
zewnętrznymi - MUSI być wywołane przed importem czegokolwiek z kodu
systemu (main_pipeline, data_download, models...), bo te moduły robią
`from db_client import supabase` przy imporcie.

- db_client   -> FakeSupabase (baza w pamięci, zero połączeń z Supabase),
- llm_comment -> brak komentarza (zero wywołań Groq).

Dzięki temu symulacja nie potrzebuje pliku .env i nie ma jak dotknąć
produkcyjnej bazy.
"""
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.fake_db import FakeSupabase  # noqa: E402

fake_supabase = FakeSupabase()

_db_client = types.ModuleType("db_client")
_db_client.supabase = fake_supabase
_db_client.STORAGE_BUCKET = "models"
sys.modules["db_client"] = _db_client

_llm_comment = types.ModuleType("llm_comment")
_llm_comment.build_llm_comment = lambda model_type, shap_values, predicted_value: None
sys.modules["llm_comment"] = _llm_comment
