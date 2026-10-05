import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "Brak skonfigurowanego GROQ_API_KEY w pliku .env!"
    )

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "openai/gpt-oss-20b"

NAIVE_STATIC_COMMENT = "Model bazowy: prognoza równa ostatniej znanej cenie złota."
NO_PREDICTORS_STATIC_COMMENT = (
    "Model nie znalazł cech, które przewidywałyby jutrzejszą zmianę ceny lepiej niż "
    "średnia historyczna - prognoza to ostatnia cena powiększona o średni dzienny zwrot."
)


def build_llm_comment(model_type: str, shap_values: dict | None, predicted_value: float) -> str:
    """
    Zwraca komentarz w języku naturalnym opisujący dzisiejszą predykcję.
    Dla modelu naiwnego (shap_values=None) - stały tekst, bez wywołania LLM
    (nie ma cech do wytłumaczenia).
    """
    if model_type == "naive":
        return NAIVE_STATIC_COMMENT
    if not shap_values:
        # Model bez predyktorów (np. Lasso, które wyzerowało wszystkie
        # współczynniki) - nie ma czego tłumaczyć, LLM niepotrzebny.
        return NO_PREDICTORS_STATIC_COMMENT

    sorted_contributions = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    contributions_text = "\n".join(
        f"- {feature}: {value:+.2f} USD" for feature, value in sorted_contributions
    )

    prompt = (
        f"Jesteś asystentem opisującym w prosty, zrozumiały sposób predykcję modelu "
        f"typu '{model_type}', przewidującego jutrzejszą cenę złota (w USD). Model "
        f"przewidział cenę {predicted_value:.2f} USD. Poniżej wkład (SHAP) poszczególnych "
        f"cech w tę predykcję, PRZELICZONY JUŻ NA DOLARY (dodatnia wartość = pchała cenę "
        f"w górę o tyle USD, ujemna = w dół o tyle USD; cechy to dzienne stopy zwrotu "
        f"instrumentów finansowych, 'actual_y' to wczorajsza stopa zwrotu samego złota):\n\n"
        f"{contributions_text}\n\n"
        f"Napisz 2-3 zdania po polsku, prostym językiem (bez żargonu technicznego), "
        f"wyjaśniające, co najbardziej wpłynęło na tę predykcję i w którą stronę. "
        f"Podawaj wkład cech w dolarach, tak jak podano wyżej, nie jako procenty ani "
        f"surowe ułamki."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("Groq zwrócił pustą odpowiedź (brak wyjątku)")
        return content
    except Exception as e:
        print(f"Błąd wywołania Groq API: {e}")
        return "Nie udało się wygenerować opisu predykcji (błąd API)."
