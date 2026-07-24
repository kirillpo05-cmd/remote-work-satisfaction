"""M6 — Streamlit UI (SPEC M6).

Contains no business logic and imports nothing from rwsat internals: every
number on every screen comes over HTTP from the API. The form is generated
from GET /schema.
"""

import os
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("RWSAT_API_URL", "http://localhost:8000")
STARTUP_HINT = (
    "The API is not reachable at `{url}`. Start the product with "
    "`docker compose up` (API on :8000, UI on :8501), or run the API alone "
    "with `uv run uvicorn rwsat.api:app --port 8000`."
)
VERDICT_COLORS = {"signal": "#2e7d32", "inconclusive": "#b8860b", "noise": "#757575"}
IMPORTANCE_SENTENCE = (
    "Permutation importance describes the fitted model, not the data, and can "
    "be positive for a model that loses to the baseline. These rows are not "
    "driver claims — factor evidence lives on the Factor Explorer screen."
)


class ApiError(Exception):
    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def _request(method: str, path: str, payload: dict | None = None) -> Any:
    try:
        response = requests.request(method, f"{API_URL}{path}", json=payload, timeout=10)
    except requests.exceptions.RequestException:
        st.error(STARTUP_HINT.format(url=API_URL))
        st.stop()
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(response.status_code, detail)
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str) -> Any:
    return _request("GET", path)


def api_post(path: str, payload: dict) -> Any:
    return _request("POST", path, payload)


def show_api_error(error: ApiError) -> None:
    st.warning(f"The API answered {error.status}: {error.detail}")


# --- Screen 1: Overview ------------------------------------------------------


def screen_overview() -> None:
    st.title("Remote Work Satisfaction Explorer")
    try:
        with st.spinner("Loading the signal report..."):
            report = api_get("/effects")
    except ApiError as error:
        show_api_error(error)
        return
    if not report["effects"]:
        st.info("The signal report is empty — no features were screened yet.")
        return

    verdict = report["dataset_verdict"]
    if verdict == "signal_present":
        st.markdown(
            f"<h2 style='color:{VERDICT_COLORS['signal']}'>Signal present</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"At least one factor shows a real, non-negligible association with "
            f"satisfaction: {report['n_signal']} of {report['n_features_tested']} "
            f"screened features earned a `signal` verdict. The Factor Explorer "
            f"shows which ones and how strong the evidence is."
        )
    else:
        st.markdown(
            f"<h2 style='color:{VERDICT_COLORS['noise']}'>No detectable signal</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"None of the {report['n_features_tested']} screened factors shows an "
            f"association with satisfaction that survives an honest statistical "
            f"check. Every factor was compared against what pure chance produces "
            f"on this same data, and none stood out. This is a finding, not a "
            f"malfunction: it protects you from spending budget on factors that "
            f"only look important."
        )
    n = max(effect["n"] for effect in report["effects"])
    st.caption(
        f"Screened on {n:,} training rows; a held-out set was kept untouched for "
        f"the model evaluation. {report['family_wise_note']}"
    )

    try:
        prediction = api_post("/predict", {"features": {}})
        st.subheader("Target distribution (training prior)")
        prior = prediction["prior_probabilities"]
        figure = go.Figure(
            go.Bar(
                x=list(prior.keys()),
                y=list(prior.values()),
                marker_color="#9e9e9e",
            )
        )
        figure.update_layout(height=280, yaxis_title="share", margin=dict(t=10))
        st.plotly_chart(figure, width="stretch")
    except ApiError:
        st.caption("Target distribution unavailable until a model artefact exists.")


# --- Screen 2: Factor Explorer -------------------------------------------------


def screen_factors() -> None:
    st.title("Factor Explorer")
    try:
        with st.spinner("Loading effects..."):
            report = api_get("/effects")
    except ApiError as error:
        show_api_error(error)
        return
    effects = report["effects"]
    if not effects:
        st.info("No effects to show yet.")
        return

    rows = [
        {
            "factor": e["feature"],
            "improvement (nats/row)": round(e["delta_logloss"], 5),
            "95% CI": f"[{e['delta_ci_low']:+.5f}, {e['delta_ci_high']:+.5f}]",
            "adjusted p": round(e["permutation_p_adj"], 3),
            "verdict": e["verdict"],
            "descriptive effect size": f"{e['native_effect_name']}={e['native_effect_size']:+.3f}",
        }
        for e in effects
    ]
    st.caption(
        "Ranked by the one number computed the same way for every factor: how "
        "much a model using only that factor improves out-of-fold prediction "
        "over always guessing the class frequencies. The descriptive effect "
        "sizes are shown for reference only — they drive neither the ranking "
        "nor the verdicts."
    )
    st.dataframe(
        rows,
        width="stretch",
        column_config={
            "verdict": st.column_config.TextColumn(help="signal / inconclusive / noise"),
        },
    )
    legend = " &nbsp; ".join(
        f"<span style='color:{color}'>&#9632; {name}</span>"
        for name, color in VERDICT_COLORS.items()
    )
    st.markdown(
        legend
        + "<br><small><b>inconclusive</b> means the evidence is insufficient to "
        "call the factor either way — that covers borderline positives as well "
        "as factors whose improvement is negative but whose permutation p is "
        "not large enough to declare noise.</small>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Signal versus noise, without words")
    chosen = st.selectbox("Factor", [e["feature"] for e in effects])
    effect = next(e for e in effects if e["feature"] == chosen)
    if not effect["null_deltas"]:
        st.info(effect["explanation"])
        return

    color = VERDICT_COLORS[effect["verdict"]]
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=effect["null_deltas"],
            nbinsx=60,
            marker_color="#b0bec5",
            name="what chance produces",
        )
    )
    figure.add_vline(
        x=effect["delta_logloss"],
        line_width=4,
        line_color=color,
        annotation_text=f"<b>observed: {effect['delta_logloss']:+.5f}</b>",
        annotation_position="top",
        annotation_font=dict(size=16, color=color),
    )
    figure.add_vline(
        x=effect["null_p95"],
        line_width=1,
        line_dash="dot",
        line_color="#607d8b",
        annotation_text="95th percentile of chance",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color="#607d8b"),
    )
    figure.update_layout(
        height=520,
        margin=dict(t=60, b=40),
        xaxis_title="out-of-fold log-loss improvement (nats per row)",
        yaxis_title=f"count of {len(effect['null_deltas'])} target shuffles",
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")
    st.markdown(
        f"The grey bars show the improvement produced by **{len(effect['null_deltas'])} "
        f"random shuffles of the target** — pure chance, on this exact data. The "
        f"thick <span style='color:{color}'>coloured line</span> is what the real "
        f"data produced. A real driver would stand far to the right of the bars.",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Verdict: {effect['verdict']}.** {effect['explanation']}")


# --- Screen 3: Predict ---------------------------------------------------------


def screen_predict() -> None:
    st.title("Predict")
    try:
        with st.spinner("Loading the form schema..."):
            contract = api_get("/schema")
    except ApiError as error:
        show_api_error(error)
        return

    with st.form("predict_form"):
        values: dict[str, Any] = {}
        columns = st.columns(2)
        for i, feature in enumerate(contract["features"]):
            with columns[i % 2]:
                if feature["kind"] == "numeric":
                    values[feature["name"]] = st.number_input(
                        feature["label"],
                        min_value=feature["min"],
                        max_value=feature["max"],
                        value=feature["default"],
                    )
                else:
                    values[feature["name"]] = st.selectbox(
                        feature["label"], feature["options"]
                    )
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return
    try:
        with st.spinner("Predicting..."):
            prediction = api_post("/predict", {"features": values})
    except ApiError as error:
        if error.status == 422 and isinstance(error.detail, list):
            for item in error.detail:
                st.warning(f"{item.get('field')}: {item.get('reason')}")
        else:
            show_api_error(error)
        return

    if prediction["confidence"] == "not_better_than_guessing":
        st.error(
            "**This prediction is not better than guessing.** "
            + (prediction["warning"] or "")
            + " Do not use it to make decisions about people."
        )

    classes = list(prediction["probabilities"].keys())
    figure = go.Figure(
        [
            go.Bar(
                name="prediction",
                x=classes,
                y=[prediction["probabilities"][c] for c in classes],
                marker_color="#1565c0",
            ),
            go.Bar(
                name="prior (no model)",
                x=classes,
                y=[prediction["prior_probabilities"][c] for c in classes],
                marker_color="#bdbdbd",
            ),
        ]
    )
    figure.update_layout(barmode="group", height=360, yaxis_title="probability")
    st.plotly_chart(figure, width="stretch")
    st.markdown(
        f"Predicted class: **{prediction['predicted_class']}** &nbsp;|&nbsp; "
        f"confidence: **{prediction['confidence']}** &nbsp;|&nbsp; "
        f"max deviation from prior: {prediction['max_deviation_from_prior']:.3f}"
    )
    if prediction["flags"]:
        st.caption("Flags: " + ", ".join(prediction["flags"]))


# --- Screen 4: Model Card --------------------------------------------------------


def screen_model_card() -> None:
    st.title("Model Card")
    try:
        with st.spinner("Loading model cards..."):
            cards = api_get("/model-card/all")
            selected = api_get("/model-card")["model_name"]
    except ApiError as error:
        show_api_error(error)
        return

    st.markdown(
        f"Selected model: **{selected}** — the simplest model whose "
        f"cross-validated log loss is statistically indistinguishable from the "
        f"best one (paired per-fold rule)."
    )
    metric_rows = []
    for name, card in cards.items():
        metric_rows.append(
            {
                "model": name + (" (selected)" if name == selected else ""),
                "CV log loss": f"{card['cv_mean']:.4f} ± {card['cv_std']:.4f}",
                "test log loss": round(card["metrics"]["log_loss"], 4),
                "baseline log loss": round(card["baseline_metrics"]["log_loss"], 4),
                "balanced accuracy": round(card["metrics"]["balanced_accuracy"], 4),
                "Brier": round(card["metrics"]["brier"], 4),
                "beats baseline": card["is_better_than_baseline"],
            }
        )
    st.dataframe(metric_rows, width="stretch")

    reference = next(iter(cards.values()))
    st.markdown(
        f"**Model-level permutation test** (gradient boosting, diagnostic): "
        f"p = {reference['permutation_test_p']:.3f}. A p-value this large means "
        f"the model trained on real labels does no better than models trained "
        f"on shuffled labels."
    )

    for name, card in cards.items():
        with st.expander(f"{name} — details"):
            for warning in card["warnings"]:
                st.warning(warning)
            if card["proportional_odds_ok"] is not None:
                st.markdown(
                    f"Proportional-odds assumption: "
                    f"{'holds' if card['proportional_odds_ok'] else 'violated'} "
                    f"(likelihood-ratio comparison against the multinomial fit)."
                )
            figure = go.Figure()
            for cls, points in card["calibration"].items():
                if points:
                    figure.add_trace(
                        go.Scatter(
                            x=[p[0] for p in points],
                            y=[p[1] for p in points],
                            mode="lines+markers",
                            name=cls,
                        )
                    )
            figure.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(dash="dash", color="#9e9e9e"), name="perfect",
                )
            )
            figure.update_layout(
                height=320,
                xaxis_title="predicted probability",
                yaxis_title="observed frequency",
                title="Calibration (one-vs-rest per class)",
            )
            st.plotly_chart(figure, width="stretch")

            if card["feature_effects"]:
                st.info(IMPORTANCE_SENTENCE)
                if not card["is_better_than_baseline"]:
                    st.caption(
                        "This model loses to the baseline, so any positive row "
                        "below reflects the model's own overfitting, not a "
                        "property of a factor."
                    )
                importance_rows = [
                    {
                        "feature": e["feature"],
                        "importance": round(e["importance_mean"], 5),
                        "95% CI": f"[{e['ci_low']:+.5f}, {e['ci_high']:+.5f}]",
                    }
                    for e in sorted(
                        card["feature_effects"], key=lambda e: -e["importance_mean"]
                    )
                ]
                st.dataframe(importance_rows, width="stretch")

    with st.expander("Data validation report"):
        try:
            st.json(api_get("/validation"))
        except ApiError as error:
            show_api_error(error)


# --- Screen 5: Methodology --------------------------------------------------------


def screen_methodology() -> None:
    st.title("Methodology")
    st.markdown(
        """
One statistic for every factor. Factors here are a mix of categories,
ordered scales and numbers, and their textbook effect sizes are not
comparable with each other. So every factor is scored the same way: **how
much does a model that knows only this factor improve its out-of-fold
predictions over always guessing the class frequencies?** (Measured in log
loss, a proper scoring rule.)

**How chance is accounted for.** For every factor the target column is
shuffled 2,000 times and the same score is recomputed each time. That builds
a picture of what pure chance produces on this data. The observed score is
compared against it; the share of shuffles that do at least as well is the
p-value (the observed run counts as one draw, so p is never exactly zero).

**Why the correction matters.** With 18 factors tested at once, chance alone
would hand roughly one of them a "significant" p-value at the 5% level. All
p-values are Benjamini-Hochberg adjusted across the full family before any
verdict is issued.

**How to read a verdict.**
- **signal** — the improvement beats chance after correction *and* its
  confidence interval clears the negligibility line. Worth acting on.
- **noise** — chance regularly does at least as well, and the interval rules
  out anything but a negligible improvement.
- **inconclusive** — neither of the above can be established. This is an
  honest state, not a hedge, and it is never collapsed into the other two.

**Known limitations.**
- The factor screen tests one factor at a time and cannot see interaction
  effects by construction; the model-level permutation test in the Model
  Card is the safeguard for that case.
- Onsite employees stay in the sample, so the target reads as satisfaction
  with the current work arrangement; conclusions about remote work
  specifically are weaker for it.
- The dataset is synthetic (5,000 generated records), and its columns show
  internal inconsistencies (e.g. respondents whose work experience exceeds
  their age minus 16), which is independent evidence the columns were
  generated separately from one another.
        """
    )


SCREENS = {
    "Overview": screen_overview,
    "Factor Explorer": screen_factors,
    "Predict": screen_predict,
    "Model Card": screen_model_card,
    "Methodology": screen_methodology,
}


def main() -> None:
    st.set_page_config(page_title="RWSAT", layout="wide")
    choice = st.sidebar.radio("Screen", list(SCREENS))
    SCREENS[choice]()


main()
