import altair as alt
import pandas as pd
import streamlit as st

from charts import ACCENT_COLOR, CATEGORY_AXIS, LABEL_STYLE, bar_y_domain, model_color_scale
from db import last_n_evaluated_cutoff, supabase
from formatting import polish_plural
from theme import apply_theme

apply_theme()

st.title("Porównanie modeli")

# Okno wyboru modelu aktywnego czytane z konfiguracji (zmienialne w panelu
# admina) - metryki niżej to osobny, stały widok 30 dni i nie muszą
# wskazywać tego samego modelu co wybór systemu.
window_resp = (
    supabase.table("pipeline_config")
    .select("value")
    .eq("key", "rolling_mape_window_days")
    .eq("active", True)
    .execute()
)
window_days = window_resp.data[0]["value"] if window_resp.data else None
selection_text = (
    f"model aktywny wybierany wg MAPE z ostatnich {int(window_days)} dni"
    if window_days is not None
    else "model aktywny wybierany wg kroczącego MAPE"
)
st.caption(f"Metryki modeli z ostatnich 30 dni · {selection_text}")

# --- Dane ---
models_resp = supabase.table("models_logs").select(
    "id, model_type, model_version, is_active, selected_features, created_at"
).execute()
models_df = pd.DataFrame(models_resp.data or [])
if models_df.empty:
    st.info("Brak jeszcze żadnych danych.")
    st.stop()

model_order = models_df.groupby("model_type")["created_at"].min().sort_values().index.tolist()

log_resp = supabase.table("system_logs").select("log_date, active_model, mape").order("log_date").execute()
system_df = pd.DataFrame(log_resp.data or [])
if system_df.empty:
    st.info("Brak jeszcze danych w system_logs.")
    st.stop()
latest_mape = system_df.iloc[-1]["mape"] or {}
active_model = system_df.iloc[-1]["active_model"]

# --- Zestawienie modeli ---
st.subheader("Zestawienie modeli")

versions_df = models_df.groupby("model_type").size().reindex(model_order)
active_rows = models_df[models_df["is_active"]].set_index("model_type")

cols = st.columns(len(model_order))
for col, model_type in zip(cols, model_order):
    active_row = active_rows.loc[model_type] if model_type in active_rows.index else None
    n_features = len(active_row["selected_features"]) if active_row is not None and active_row["selected_features"] else 0
    n_versions_text = polish_plural(int(versions_df[model_type]), "wersja", "wersje", "wersji")
    n_features_text = polish_plural(n_features, "cecha", "cechy", "cech")

    is_active = model_type == active_model
    border_color = "#22C55E" if is_active else "#2A2F3A"
    background = "rgba(34, 197, 94, 0.08)" if is_active else "transparent"
    col.markdown(
        f"""
        <div style='border:2px solid {border_color}; border-radius:8px; padding:0.9rem 1rem;
                    background:{background};'>
            <div style='font-weight:600;'>{model_type}</div>
            <div style='color:#94a3b8; font-size:0.85rem; margin-top:0.2rem;'>{n_versions_text} · {n_features_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.divider()

# --- Metryki ---
st.subheader("Metryki")
st.caption("(ostatnie 30 dni)")

id_to_type = dict(zip(models_df["id"], models_df["model_type"]))
# "30 dni" = 30 ostatnich dni z FAKTYCZNIE ewaluowaną predykcją
recent_cutoff, _ = last_n_evaluated_cutoff(30)
recent_resp = (
    supabase.table("model_predictions")
    .select("target_date, model_id, predicted_value, actual_value, error_value")
    .eq("status", "evaluated")
    .gte("target_date", recent_cutoff)
    .execute()
)
recent_df = pd.DataFrame(recent_resp.data or [])

if not recent_df.empty:
    recent_df["target_date"] = pd.to_datetime(recent_df["target_date"])
    recent_df["model_type"] = recent_df["model_id"].map(id_to_type)

    actual_series = (
        recent_df.drop_duplicates("target_date").sort_values("target_date")[["target_date", "actual_value"]]
        .reset_index(drop=True)
    )
    actual_series["prev_actual"] = actual_series["actual_value"].shift(1)
    prev_actual_map = dict(zip(actual_series["target_date"], actual_series["prev_actual"]))
    recent_df["prev_actual"] = recent_df["target_date"].map(prev_actual_map)

    metric_rows = []
    for model_type in model_order:
        sub = recent_df[recent_df["model_type"] == model_type]
        if sub.empty:
            continue
        mae = sub["error_value"].abs().mean()
        rmse = (sub["error_value"] ** 2).mean() ** 0.5

        hit_sub = sub.dropna(subset=["prev_actual"])
        if not hit_sub.empty:
            actual_dir = (hit_sub["actual_value"] - hit_sub["prev_actual"]) > 0
            pred_dir = (hit_sub["predicted_value"] - hit_sub["prev_actual"]) > 0
            hit_rate = (actual_dir == pred_dir).mean() * 100
        else:
            hit_rate = None

        metric_rows.append({
            "model": model_type,
            "MAPE": latest_mape.get(model_type),
            "MAE": mae,
            "RMSE": rmse,
            "Trafność": hit_rate,
        })

    metrics_df = pd.DataFrame(metric_rows)

    def metric_chart(value_col: str, title: str, fmt: str, higher_is_better: bool = False) -> alt.LayerChart:
        sub_df = metrics_df.dropna(subset=[value_col]).copy()
        winner = sub_df["model"][sub_df[value_col].idxmax() if higher_is_better else sub_df[value_col].idxmin()]
        # Reszta słupków przygaszona (opacity), zwycięzca w pełnej sile +
        # złota obwódka
        sub_df["is_winner"] = sub_df["model"] == winner

        bars = (
            alt.Chart(sub_df)
            .mark_bar()
            .encode(
                x=alt.X("model:N", title=None, sort=model_order, axis=CATEGORY_AXIS),
                y=alt.Y(f"{value_col}:Q", title=title, scale=alt.Scale(domain=bar_y_domain(sub_df[value_col]))),
                color=alt.Color("model:N", scale=model_color_scale(model_order), legend=None),
                opacity=alt.condition("datum.is_winner", alt.value(1.0), alt.value(0.4)),
                stroke=alt.condition("datum.is_winner", alt.value(ACCENT_COLOR), alt.value(None)),
                strokeWidth=alt.condition("datum.is_winner", alt.value(2.5), alt.value(0)),
                tooltip=[
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip(f"{value_col}:Q", title=title, format=fmt),
                ],
            )
        )
        labels = (
            alt.Chart(sub_df)
            .mark_text(**LABEL_STYLE)
            .encode(
                x=alt.X("model:N", sort=model_order),
                y=f"{value_col}:Q",
                text=alt.Text(f"{value_col}:Q", format=fmt),
                opacity=alt.condition("datum.is_winner", alt.value(1.0), alt.value(0.55)),
                tooltip=alt.value(None),
            )
        )
        return (bars + labels).properties(height=240)

    st.caption("Najlepszy wynik w danej kategorii jest wyróżniony")
    row1 = st.columns(2)
    with row1[0]:
        st.altair_chart(metric_chart("MAPE", "MAPE (%)", ".2f"), width="stretch")
    with row1[1]:
        st.altair_chart(metric_chart("MAE", "MAE ($)", ".2f"), width="stretch")

    row2 = st.columns(2)
    with row2[0]:
        st.altair_chart(metric_chart("RMSE", "RMSE ($)", ".2f"), width="stretch")
    with row2[1]:
        st.altair_chart(metric_chart("Trafność", "Trafność kierunku (%)", ".1f", higher_is_better=True), width="stretch")
else:
    st.info("Brak ocenionych predykcji z ostatnich 30 dni.")
