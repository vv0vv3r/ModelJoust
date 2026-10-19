import altair as alt
import pandas as pd
import streamlit as st

from charts import (
    ACTUAL_COLOR,
    MODEL_COLORS,
    adaptive_time_axis,
    highlight_best_worst,
    model_color_scale,
    padded_domain,
)
from db import fetch_all, last_n_evaluated_cutoff, supabase
from theme import apply_theme

apply_theme()

st.title("Historia predykcji i jakość")
st.caption("Predykcje vs rzeczywiste wartości w czasie")

RANGE_OPTIONS = ["10 dni", "30 dni", "90 dni", "180 dni", "365 dni", "wszystkie"]
zakres = st.segmented_control("Zakres (N ostatnich dni sesyjnych)", RANGE_OPTIONS, default="30 dni")
zakres = zakres or "30 dni"

requested_axis_start = None
if zakres == "wszystkie":
    cutoff = None
else:
    # "N dni" = N dni z FAKTYCZNIE ewaluowaną predykcją
    days = int(zakres.split(" ")[0])
    cutoff, found_count = last_n_evaluated_cutoff(days)

    if found_count < days:
        # Historii jest mniej niż N dni - lewa krawędź osi i tak pokazuje
        # PEŁNE żądane okno (licząc kalendarzowo), żeby było widać, że
        # wybrano więcej niż faktycznie mamy. Bezpieczne: N dni
        # kalendarzowych zawsze obejmuje co najmniej N dni z ewaluacją, więc
        # dane nigdy nie wypadną poza tak ustawioną oś.
        requested_axis_start = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)

# Kolejność modeli spójna z resztą apki (Retreningi) - wg pierwszego treningu.
models_resp = supabase.table("models_logs").select("id, model_type, created_at").execute()
models_df = pd.DataFrame(models_resp.data or [])
if models_df.empty:
    st.info("Brak jeszcze żadnych danych.")
    st.stop()

id_to_type = dict(zip(models_df["id"], models_df["model_type"]))
model_order = (
    models_df.groupby("model_type")["created_at"].min().sort_values().index.tolist()
)

def _pred_query():
    query = (
        supabase.table("model_predictions")
        .select("target_date, model_id, predicted_value, actual_value, error_value")
        .eq("status", "evaluated")
    )
    if cutoff:
        query = query.gte("target_date", cutoff)
    return query.order("target_date").order("prediction_id")


pred_rows = fetch_all(_pred_query)

if not pred_rows:
    st.info("Brak jeszcze ocenionych predykcji w wybranym zakresie.")
    st.stop()

df = pd.DataFrame(pred_rows)
df["target_date"] = pd.to_datetime(df["target_date"])
df["model_type"] = df["model_id"].map(id_to_type)

# Widoczność kropek na liniach zależy od FAKTYCZNEGO rozstrzału dat w
# przefiltrowanych danych, nie od wybranej etykiety zakresu - dzięki temu
# "wszystkie" też dostanie kropki, dopóki historia jest krótka, i straci
# je automatycznie, gdy urośnie ponad 90 dni, bez specjalnego przypadku.
actual_span_days = (df["target_date"].max() - df["target_date"].min()).days
show_points = actual_span_days < 90

# Lewa krawędź: requested_axis_start (patrz wyżej) gdy historii brakuje,
# inaczej ciasno dopasowana do danych (bo wtedy i tak się z nimi pokrywa).
# Prawa krawędź: zawsze ciasno dopasowana - nie sięga do "dziś", żeby nie
# było pustej przestrzeni za weekendy/dzisiejszą (jeszcze pending) predykcję.
axis_start = requested_axis_start if requested_axis_start is not None else df["target_date"].min()
axis_domain = [axis_start, df["target_date"].max()]

# --- Metryki błędu ---
st.subheader("Metryki błędu")
st.caption("(W wybranym zakresie)")

# Trafność kierunku: czy przewidziany kierunek zmiany (względem POPRZEDNIEGO
# dnia sesyjnego) zgadza się z rzeczywistym. Punkt odniesienia to ostatnia
# znana cena, wspólna dla wszystkich modeli na dany target_date - odtworzona
# z samych evaluowanych predykcji (przesunięcie o 1 dzień sesyjny), bez
# potrzeby osobnego zapytania do raw_data.
actual_series = (
    df.drop_duplicates("target_date").sort_values("target_date")[["target_date", "actual_value"]]
    .reset_index(drop=True)
)
actual_series["prev_actual"] = actual_series["actual_value"].shift(1)
prev_actual_map = dict(zip(actual_series["target_date"], actual_series["prev_actual"]))
df["prev_actual"] = df["target_date"].map(prev_actual_map)

metrics_rows = []
for model_type in model_order:
    sub = df[df["model_type"] == model_type]
    if sub.empty:
        continue
    mape = (sub["error_value"].abs() / sub["actual_value"].abs()).mean() * 100
    mae = sub["error_value"].abs().mean()
    rmse = (sub["error_value"] ** 2).mean() ** 0.5
    bias = sub["error_value"].mean()

    hit_sub = sub.dropna(subset=["prev_actual"])
    if not hit_sub.empty:
        actual_dir = (hit_sub["actual_value"] - hit_sub["prev_actual"]) > 0
        pred_dir = (hit_sub["predicted_value"] - hit_sub["prev_actual"]) > 0
        hit_rate = (actual_dir == pred_dir).mean() * 100
    else:
        hit_rate = None

    metrics_rows.append({
        "Model": model_type,
        "MAPE (%)": mape,
        "MAE ($)": mae,
        "RMSE ($)": rmse,
        "Bias ($)": bias,
        "Trafność kierunku (%)": hit_rate,
    })

metrics_df = pd.DataFrame(metrics_rows)
st.dataframe(
    metrics_df.style.format({
        "MAPE (%)": "{:.2f}",
        "MAE ($)": "{:.2f}",
        "RMSE ($)": "{:.2f}",
        "Bias ($)": "{:+.2f}",
        "Trafność kierunku (%)": "{:.1f}",
    }, na_rep="—").apply(highlight_best_worst, axis=0),
    hide_index=True,
    width="stretch",
)

st.divider()

# --- Kroczące MAPE ---
st.subheader("Kroczące MAPE")
mape_query = supabase.table("system_logs").select("log_date, mape")
if cutoff:
    mape_query = mape_query.gte("log_date", cutoff)
mape_resp = mape_query.order("log_date").execute()

mape_rows = []
for row in mape_resp.data or []:
    for model_type, value in (row.get("mape") or {}).items():
        if value is not None:
            mape_rows.append({"log_date": row["log_date"], "model_type": model_type, "MAPE": value})

if mape_rows:
    mape_df = pd.DataFrame(mape_rows)
    mape_df["log_date"] = pd.to_datetime(mape_df["log_date"])
    mape_chart = (
        alt.Chart(mape_df)
        .mark_line(point=show_points)
        .encode(
            x=alt.X(
                "log_date:T",
                axis=adaptive_time_axis(axis_start),
                scale=alt.Scale(domain=axis_domain, nice=False),
            ),
            y=alt.Y("MAPE:Q", title="MAPE (%)", scale=alt.Scale(domain=padded_domain(mape_df["MAPE"]))),
            color=alt.Color("model_type:N", title=None, scale=model_color_scale(model_order)),
            tooltip=[
                alt.Tooltip("log_date:T", title="Data"),
                alt.Tooltip("model_type:N", title="Model"),
                alt.Tooltip("MAPE:Q", title="MAPE", format=".2f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(mape_chart, width="stretch")
else:
    st.info("Brak danych MAPE w wybranym zakresie.")

st.divider()

# --- Predykcja vs rzeczywista wartość ---
st.subheader("Predykcja vs rzeczywista wartość")

pred_lines = df[["target_date", "model_type", "predicted_value"]].rename(
    columns={"model_type": "seria", "predicted_value": "wartość"}
)
actual_lines = (
    df.drop_duplicates("target_date")[["target_date", "actual_value"]]
    .rename(columns={"actual_value": "wartość"})
)
actual_lines["seria"] = "Zamknięcie"
line_df = pd.concat([pred_lines, actual_lines[["target_date", "seria", "wartość"]]], ignore_index=True)
line_df["is_actual"] = line_df["seria"] == "Zamknięcie"

price_chart = (
    alt.Chart(line_df)
    .mark_line(point=show_points)
    .encode(
        x=alt.X(
            "target_date:T",
            axis=adaptive_time_axis(axis_start),
            scale=alt.Scale(domain=axis_domain, nice=False),
        ),
        y=alt.Y("wartość:Q", title="Cena ($)", scale=alt.Scale(domain=padded_domain(line_df["wartość"]))),
        color=alt.Color(
            "seria:N",
            title=None,
            scale=model_color_scale(model_order, extra={"Zamknięcie": ACTUAL_COLOR}),
        ),
        strokeWidth=alt.StrokeWidth(
            "is_actual:N", scale=alt.Scale(domain=[True, False], range=[3, 1.5]), legend=None
        ),
        tooltip=[
            alt.Tooltip("target_date:T", title="Data"),
            alt.Tooltip("seria:N", title="Seria"),
            alt.Tooltip("wartość:Q", title="Wartość", format="$.2f"),
        ],
    )
    .properties(height=340)
)
st.altair_chart(price_chart, width="stretch")

st.divider()

# --- Błąd w czasie ---
st.subheader("Błąd w czasie")

zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#64748b", strokeDash=[4, 4]).encode(y="y:Q")
error_time_chart = (
    alt.Chart(df)
    .mark_line(point=show_points)
    .encode(
        x=alt.X(
            "target_date:T",
            axis=adaptive_time_axis(axis_start),
            scale=alt.Scale(domain=axis_domain, nice=False),
        ),
        y=alt.Y("error_value:Q", title="Błąd ($)", scale=alt.Scale(domain=padded_domain(df["error_value"]))),
        color=alt.Color("model_type:N", title=None, scale=model_color_scale(model_order)),
        tooltip=[
            alt.Tooltip("target_date:T", title="Data"),
            alt.Tooltip("model_type:N", title="Model"),
            alt.Tooltip("error_value:Q", title="Błąd", format="+.2f"),
        ],
    )
)
st.altair_chart((zero_rule + error_time_chart).properties(height=340), width="stretch")

st.divider()

# --- Rozkład błędów ---
st.subheader("Rozkład błędów")

n_bins = 15
df["bin"] = pd.cut(df["error_value"], bins=n_bins)
df["bin_center"] = df["bin"].apply(lambda b: (b.left + b.right) / 2)
df["bin_label"] = df["bin"].apply(lambda b: f"${b.left:.2f} → ${b.right:.2f}")

hist_df = (
    df.groupby(["model_type", "bin_center", "bin_label"], observed=True)
    .size()
    .reset_index(name="liczba")
)

model_hist_colors = dict(zip(model_order, MODEL_COLORS))
hist_cols = st.columns(len(model_order))
for col, model_type in zip(hist_cols, model_order):
    sub_hist = hist_df[hist_df["model_type"] == model_type]
    small_hist = (
        alt.Chart(sub_hist)
        .mark_bar(color=model_hist_colors[model_type])
        .encode(
            x=alt.X("bin_center:Q", title="Błąd ($)"),
            y=alt.Y("liczba:Q", title="Liczba"),
            tooltip=[
                alt.Tooltip("bin_label:N", title="Zakres"),
                alt.Tooltip("liczba:Q", title="Liczba"),
            ],
        )
        .properties(height=300, title=model_type)
    )
    with col:
        st.altair_chart(small_hist, width="stretch")
