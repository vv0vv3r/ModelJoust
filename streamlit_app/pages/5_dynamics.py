import altair as alt
import pandas as pd
import streamlit as st

from charts import ACCENT_COLOR, CATEGORY_AXIS, GRID_STYLE, LABEL_STYLE, bar_y_domain, highlight_best_worst, model_color_scale
from db import fetch_all, last_n_evaluated_cutoff, supabase
from theme import apply_theme

apply_theme()

st.title("Dynamika: Champion / Challenger")
st.caption("Jak zachowuje się mechanizm wyboru aktywnego modelu, i czy dynamiczne przełączanie jest lepsze od trzymania się jednego modelu")

RANGE_OPTIONS = ["30 dni", "90 dni", "180 dni", "365 dni", "wszystkie"]
zakres = st.segmented_control("Zakres (N ostatnich dni sesyjnych)", RANGE_OPTIONS, default="180 dni")
zakres = zakres or "180 dni"

# --- Dane wspólne, przefiltrowane wybranym zakresem ---
models_resp = supabase.table("models_logs").select("id, model_type, created_at").execute()
models_df = pd.DataFrame(models_resp.data or [])
if models_df.empty:
    st.info("Brak jeszcze żadnych danych.")
    st.stop()

id_to_type = dict(zip(models_df["id"], models_df["model_type"]))
model_order = models_df.groupby("model_type")["created_at"].min().sort_values().index.tolist()

log_resp = supabase.table("system_logs").select("log_date, active_model, mape").order("log_date").execute()
system_df_full = pd.DataFrame(log_resp.data or [])
if system_df_full.empty:
    st.info("Brak jeszcze danych w system_logs.")
    st.stop()
system_df_full["log_date"] = pd.to_datetime(system_df_full["log_date"])

pred_rows = fetch_all(
    lambda: supabase.table("model_predictions")
    .select("target_date, model_id, predicted_value, actual_value, error_value")
    .eq("status", "evaluated")
    .order("prediction_id")
)
pred_df_full = pd.DataFrame(pred_rows)
pred_df_full["target_date"] = pd.to_datetime(pred_df_full["target_date"])
pred_df_full["model_type"] = pred_df_full["model_id"].map(id_to_type)

if zakres == "wszystkie":
    system_df = system_df_full
    pred_df = pred_df_full
else:
    # "N dni" = N dni z FAKTYCZNIE ewaluowaną predykcją
    days = int(zakres.split(" ")[0])
    cutoff_str, _ = last_n_evaluated_cutoff(days)
    cutoff_date = pd.Timestamp(cutoff_str) if cutoff_str else system_df_full["log_date"].min()
    system_df = system_df_full[system_df_full["log_date"] >= cutoff_date]
    pred_df = pred_df_full[pred_df_full["target_date"] >= cutoff_date]

# --- Dynamiczne przełączanie vs jeden stały model ---
st.subheader("Dynamiczne przełączanie vs jeden stały model")
st.caption(
    "Średnie MAPE w wybranym zakresie — dla każdego modelu, gdyby był aktywny przez cały "
    "okres, zestawione z MAPE wygenerowanym przez predykcje aktywnego danego dnia modelu."
)

SYSTEM_LABEL = "system"

realized_df = system_df.merge(
    pred_df, left_on=["log_date", "active_model"], right_on=["target_date", "model_type"]
)

comparison_rows = []
for model_type in model_order:
    sub = pred_df[pred_df["model_type"] == model_type]
    if sub.empty:
        continue
    static_mape = (sub["error_value"].abs() / sub["actual_value"].abs()).mean() * 100
    comparison_rows.append({"model": model_type, "MAPE": static_mape})

if not realized_df.empty:
    realized_mape = (realized_df["error_value"].abs() / realized_df["actual_value"].abs()).mean() * 100
    comparison_rows.append({"model": SYSTEM_LABEL, "MAPE": realized_mape})

if comparison_rows:
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_order = model_order + [SYSTEM_LABEL]

    # Delta vs system - tylko dla słupków modeli statycznych (system
    # porównywany sam ze sobą nie ma sensu). Zielony = ta wartość biła system
    # (niższe MAPE),
    # czerwony = przegrała z systemem (wyższe MAPE).
    system_rows = comparison_df.loc[comparison_df["model"] == SYSTEM_LABEL, "MAPE"]
    if not system_rows.empty:
        system_mape_value = float(system_rows.iloc[0])
        delta_df = comparison_df[comparison_df["model"] != SYSTEM_LABEL].copy()
        delta_df["delta_vs_system"] = delta_df["MAPE"] - system_mape_value
        delta_df["delta_label"] = delta_df["delta_vs_system"].apply(lambda v: f"({v:+.2f})")
    else:
        delta_df = pd.DataFrame(columns=["model", "MAPE", "delta_vs_system", "delta_label"])

    comp_bars = (
        alt.Chart(comparison_df)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None, sort=comparison_order, axis=CATEGORY_AXIS),
            y=alt.Y(
                "MAPE:Q",
                title="Średnie MAPE (%)",
                scale=alt.Scale(domain=bar_y_domain(comparison_df["MAPE"])),
            ),
            color=alt.condition(
                alt.datum.model == SYSTEM_LABEL,
                alt.value("#C9A961"),
                alt.value("#64748b"),
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("MAPE:Q", title="MAPE", format=".2f"),
            ],
        )
    )
    # Niezależny wykres, nie comp_bars.mark_text() - dziedziczenie enkodowań
    # z comp_bars przenosiłoby na tekst też jego warunkowy kolor złoto/szary.
    comp_labels = (
        alt.Chart(comparison_df)
        .mark_text(**LABEL_STYLE)
        .encode(
            x=alt.X("model:N", sort=comparison_order),
            y="MAPE:Q",
            text=alt.Text("MAPE:Q", format=".2f"),
            tooltip=alt.value(None),
        )
    )
    layers = [comp_bars, comp_labels]
    if not delta_df.empty:
        delta_labels = (
            alt.Chart(delta_df)
            .mark_text(dy=-28, fontSize=12, fontWeight="bold")
            .encode(
                x=alt.X("model:N", sort=comparison_order),
                y="MAPE:Q",
                text="delta_label:N",
                color=alt.condition("datum.delta_vs_system < 0", alt.value("#22C55E"), alt.value("#EF4444")),
                tooltip=alt.value(None),
            )
        )
        layers.append(delta_labels)
    st.caption("W nawiasie: różnica MAPE względem systemu (zielony = bije system, czerwony = przegrywa z systemem)")
    st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")

    # --- Pełne zestawienie metryk (MAPE + MAE/RMSE/Bias/Trafność) ---
    st.caption("Pełne zestawienie metryk")

    actual_series = (
        pred_df.drop_duplicates("target_date").sort_values("target_date")[["target_date", "actual_value"]]
        .reset_index(drop=True)
    )
    actual_series["prev_actual"] = actual_series["actual_value"].shift(1)
    prev_actual_map = dict(zip(actual_series["target_date"], actual_series["prev_actual"]))

    def _hit_rate(sub: pd.DataFrame) -> float | None:
        hit_sub = sub.dropna(subset=["prev_actual"])
        if hit_sub.empty:
            return None
        actual_dir = (hit_sub["actual_value"] - hit_sub["prev_actual"]) > 0
        pred_dir = (hit_sub["predicted_value"] - hit_sub["prev_actual"]) > 0
        return (actual_dir == pred_dir).mean() * 100

    full_metrics_rows = []
    for model_type in model_order:
        sub = pred_df[pred_df["model_type"] == model_type].copy()
        if sub.empty:
            continue
        sub["prev_actual"] = sub["target_date"].map(prev_actual_map)
        full_metrics_rows.append({
            "Model": model_type,
            "MAPE (%)": (sub["error_value"].abs() / sub["actual_value"].abs()).mean() * 100,
            "MAE ($)": sub["error_value"].abs().mean(),
            "RMSE ($)": (sub["error_value"] ** 2).mean() ** 0.5,
            "Bias ($)": sub["error_value"].mean(),
            "Trafność kierunku (%)": _hit_rate(sub),
        })

    if not realized_df.empty:
        realized_df = realized_df.copy()
        realized_df["prev_actual"] = realized_df["target_date"].map(prev_actual_map)
        full_metrics_rows.append({
            "Model": SYSTEM_LABEL,
            "MAPE (%)": realized_mape,
            "MAE ($)": realized_df["error_value"].abs().mean(),
            "RMSE ($)": (realized_df["error_value"] ** 2).mean() ** 0.5,
            "Bias ($)": realized_df["error_value"].mean(),
            "Trafność kierunku (%)": _hit_rate(realized_df),
        })

    def highlight_system_row(row: pd.Series) -> list[str]:
        # System nie jest "jeszcze jednym kandydatem" tylko realną strategią,
        # która faktycznie działała - odznaczona ramką i pogrubieniem, nie
        # kolorem (żeby nie wchodzić w konflikt z zielony/czerwony best/worst).
        if row["Model"] == SYSTEM_LABEL:
            return [f"border-top: 2px solid {ACCENT_COLOR}; font-weight: 600;"] * len(row)
        return ["" for _ in row]

    full_metrics_df = pd.DataFrame(full_metrics_rows)
    st.dataframe(
        full_metrics_df.style.format({
            "MAPE (%)": "{:.2f}",
            "MAE ($)": "{:.2f}",
            "RMSE ($)": "{:.2f}",
            "Bias ($)": "{:+.2f}",
            "Trafność kierunku (%)": "{:.1f}",
        }, na_rep="—").apply(highlight_best_worst, axis=0).apply(highlight_system_row, axis=1),
        hide_index=True,
        width="stretch",
    )
else:
    st.info("Brak danych w wybranym zakresie.")

st.divider()

# --- Ile dni każdy model był aktywny ---
st.subheader("Ile dni każdy model był aktywny")

active_days = system_df["active_model"].value_counts().reindex(model_order).fillna(0).astype(int)
active_days_df = active_days.reset_index()
active_days_df.columns = ["model", "dni"]

days_bars = (
    alt.Chart(active_days_df)
    .mark_bar()
    .encode(
        x=alt.X("model:N", title=None, sort=model_order, axis=CATEGORY_AXIS),
        y=alt.Y(
            "dni:Q",
            title="Liczba dni aktywności",
            axis=alt.Axis(tickMinStep=1, format="d"),
            scale=alt.Scale(domain=bar_y_domain(active_days_df["dni"])),
        ),
        color=alt.Color("model:N", scale=model_color_scale(model_order), legend=None),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("dni:Q", title="Dni aktywności"),
        ],
    )
)
# Niezależny wykres, nie days_bars.mark_text()
days_labels = (
    alt.Chart(active_days_df)
    .mark_text(**LABEL_STYLE)
    .encode(
        x=alt.X("model:N", sort=model_order),
        y="dni:Q",
        text="dni:Q",
        tooltip=alt.value(None),
    )
)
st.altair_chart((days_bars + days_labels).properties(height=280), width="stretch")

st.divider()

# --- Pasek czasowy aktywności ---
st.subheader("Pasek czasowy aktywności")

timeline_strip = (
    alt.Chart(system_df)
    .mark_bar(height=40)
    .encode(
        x=alt.X("log_date:O", title=None, axis=alt.Axis(labels=False, ticks=False, **GRID_STYLE)),
        color=alt.Color(
            "active_model:N",
            title=None,
            scale=model_color_scale(model_order),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("log_date:T", title="Data"),
            alt.Tooltip("active_model:N", title="Aktywny model"),
        ],
    )
    .properties(height=90)
)
st.altair_chart(timeline_strip, width="stretch")

st.divider()

# --- Korelacja błędów między modelami ---
st.subheader("Korelacja błędów między modelami")
st.caption("Czy modele mylą się razem, czy niezależnie.")

pivot = pred_df.pivot_table(index="target_date", columns="model_type", values="error_value")
available_models = [m for m in model_order if m in pivot.columns]
if len(available_models) >= 2:
    corr = pivot[available_models].corr()
    corr.index.name = "model_1"
    corr_long = corr.reset_index().melt(id_vars="model_1", var_name="model_2", value_name="corr")

    heatmap = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("model_1:N", title=None, sort=available_models),
            y=alt.Y("model_2:N", title=None, sort=available_models),
            color=alt.Color(
                "corr:Q", title="Korelacja", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)
            ),
            tooltip=[
                alt.Tooltip("model_1:N", title="Model A"),
                alt.Tooltip("model_2:N", title="Model B"),
                alt.Tooltip("corr:Q", title="Korelacja", format=".2f"),
            ],
        )
    )
    heatmap_labels = heatmap.mark_text(fontSize=13, fontWeight="bold").encode(
        text=alt.Text("corr:Q", format=".2f"),
        # Kontrast dopasowany do skali redblue - skrajne (nasycone) komórki
        # dostają biały tekst, środkowe (jasne, blisko 0) czarny.
        color=alt.condition("abs(datum.corr) > 0.45", alt.value("#FFFFFF"), alt.value("#1A1A1A")),
    )
    st.altair_chart((heatmap + heatmap_labels).properties(height=300), width="stretch")
else:
    st.info("Za mało modeli ze wspólnymi ocenionymi predykcjami do policzenia korelacji.")
