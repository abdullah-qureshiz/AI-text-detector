"""Browser-based interface for the AI text detector.

Run with: streamlit run streamlit_app.py
"""
import csv
import io
import json
from datetime import datetime

import streamlit as st

from config import Config
from detectors import MLTextDetector, RuleBasedDetector, TextPreprocessor


SAMPLE_TEXT = (
    "The community garden opened its gates on a bright Saturday morning. "
    "Neighbors carried seedlings, borrowed tools, and stories about what they "
    "hoped to grow. One child carefully labeled every pot, while two friends "
    "argued cheerfully about where the tomatoes should go. By lunchtime, the "
    "empty beds were full of color, and someone had put a kettle on. The work "
    "was not finished, but nobody seemed in a hurry to leave. They made a plan "
    "to meet again next weekend and see what the rain had done."
)

st.set_page_config(
    page_title="Signal — AI Text Detector",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: light;
        --ink: #16231d;
        --muted: #536159;
        --paper: #f3f6f2;
        --card: #ffffff;
        --line: #d7e0d8;
        --green: #246548;
        --green-dark: #194c36;
        --mint: #e6f3eb;
        --ai-ink: #814411;
        --ai-bg: #fff3e5;
        --disagree-ink: #594186;
        --disagree-bg: #f2edfa;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    [data-testid="stHeader"] { background: rgba(243,246,242,.96); }
    [data-testid="stSidebar"] { background: #eaf0e9; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; }
    h1 { font-size: clamp(2.2rem, 5vw, 3.5rem) !important; line-height: 1.05 !important; }
    h2 { font-size: 1.55rem !important; }
    p, label, [data-testid="stMarkdownContainer"] { color: var(--ink); }
    .brand { font: 700 1.1rem 'Manrope', sans-serif; letter-spacing: -.04em; }
    .brand-mark {
        display: inline-flex; width: 30px; height: 30px; margin-right: 8px;
        align-items: center; justify-content: center; border-radius: 9px;
        color: #ffffff; background: var(--green-dark); vertical-align: middle;
    }
    .eyebrow {
        color: var(--green-dark); font: 500 .72rem 'DM Mono', monospace;
        letter-spacing: .12em; text-transform: uppercase;
    }
    .hero { padding: .5rem 0 1.6rem; }
    .subtle { color: var(--muted); font-size: 1.02rem; max-width: 680px; }
    .section-label {
        color: var(--muted); font: 500 .7rem 'DM Mono', monospace;
        letter-spacing: .1em; text-transform: uppercase; margin-bottom: .65rem;
    }
    .surface {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 18px; padding: 1.25rem 1.4rem;
        box-shadow: 0 6px 20px rgba(24, 48, 34, .055);
        transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
    }
    .surface:hover {
        border-color: #b7c9bb;
        box-shadow: 0 10px 26px rgba(24, 48, 34, .09);
        transform: translateY(-2px);
    }
    .result-title { font: 700 1.1rem 'Manrope', sans-serif; margin-bottom: .2rem; }
    .result-caption { color: var(--muted); font-size: .86rem; }
    .pill {
        display: inline-block; border-radius: 999px; padding: .34rem .7rem;
        color: var(--green-dark); background: var(--mint); border: 1px solid #c5dfce;
        font: 600 .7rem 'DM Mono', monospace; text-transform: uppercase;
        letter-spacing: .04em;
    }
    .pill--ai { color: var(--ai-ink); background: var(--ai-bg); border-color: #f0d1ad; }
    .pill--disagree { color: var(--disagree-ink); background: var(--disagree-bg); border-color: #d9cbed; }
    .result-card { border-left: 4px solid var(--green); }
    .result-card--ai { border-left-color: #a25b18; }
    .result-card--disagree { border-left-color: #7053a4; }
    .warning-note {
        border: 1px solid #e6d3ad; background: #fff8eb; color: #654516;
        border-radius: 12px; padding: .85rem 1rem; font-size: .88rem;
    }
    .metric-strip {
        background: var(--card); border: 1px solid var(--line); border-radius: 14px;
        padding: .8rem 1rem; color: var(--muted); font-size: .84rem;
    }
    .metric-strip strong { color: var(--ink); font: 600 .95rem 'DM Mono', monospace; }
    div[data-testid="stTextArea"] textarea {
        border: 1px solid #bdcbbf; border-radius: 14px; background: var(--card);
        color: var(--ink); caret-color: var(--green-dark);
        font: 400 .98rem/1.7 'DM Sans', sans-serif; padding: 1rem;
        transition: border-color .16s ease, box-shadow .16s ease;
    }
    div[data-testid="stTextArea"] textarea::placeholder { color: #66746b; opacity: 1; }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--green-dark); box-shadow: 0 0 0 3px rgba(36, 101, 72, .16);
    }
    div.stButton > button, [data-testid="stDownloadButton"] button {
        border: 1px solid #aebfb1; color: var(--ink); background: #ffffff;
        border-radius: 10px; min-height: 2.65rem; font-weight: 600;
        transition: color .16s ease, background .16s ease, border-color .16s ease, box-shadow .16s ease, transform .16s ease;
    }
    div.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
        color: var(--green-dark); background: #edf5ef; border-color: #6c987c;
        transform: translateY(-1px);
    }
    div.stButton > button:focus-visible, [data-testid="stDownloadButton"] button:focus-visible {
        outline: 3px solid rgba(36, 101, 72, .35); outline-offset: 2px;
    }
    div.stButton > button:active, [data-testid="stDownloadButton"] button:active { transform: translateY(0); }
    div.stButton > button[kind="primary"] {
        color: #ffffff; background: var(--green-dark); border-color: var(--green-dark);
        border-radius: 11px; min-height: 2.85rem; font-weight: 700;
    }
    div.stButton > button[kind="primary"]:hover {
        color: #ffffff; background: #123d2a; border-color: #123d2a;
        box-shadow: 0 5px 12px rgba(25, 76, 54, .2);
    }
    div[data-testid="stTabs"] button {
        color: var(--muted); font-weight: 600; transition: color .15s ease;
    }
    div[data-testid="stTabs"] button:hover { color: var(--green-dark); }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--green-dark); border-bottom-color: var(--green-dark);
    }
    div[data-testid="stProgress"] > div { background: #dce7de; border-radius: 99px; }
    div[data-testid="stProgress"] > div > div { background: var(--green-dark); border-radius: 99px; }
    [data-testid="stMetric"] {
        color: var(--ink); background: #ffffff; padding: .85rem;
        border-radius: 12px; border: 1px solid var(--line);
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stExpander"] {
        background: #ffffff; border: 1px solid var(--line); border-radius: 12px;
    }
    [data-testid="stExpander"] summary:hover { color: var(--green-dark); }
    [data-testid="stSlider"] [role="slider"] { background: var(--green-dark); border-color: #ffffff; }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: var(--green-dark); }
    [data-testid="stSidebar"] hr { border-color: #ced9d0; }
    @media (max-width: 760px) {
        .block-container { padding: 1.25rem 1rem 3rem; }
        .hero { padding-bottom: 1rem; }
        .surface { padding: 1rem; }
        .metric-strip { line-height: 1.8; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def word_count(text):
    return len(TextPreprocessor.words(text))


def result_rows(results):
    rows = []
    for detector, result in results.items():
        if "error" not in result:
            rows.append(
                {
                    "Detector": detector,
                    "Result": result["label"],
                    "Confidence": f'{result["confidence"]}%',
                    "AI likelihood": f'{result["ai_probability"] * 100:.1f}%',
                }
            )
    return rows


def make_history_csv(history):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["timestamp", "threshold", "detector", "result", "confidence", "ai_probability", "text"]
    )
    for entry in history:
        for detector, result in entry["results"].items():
            if "error" not in result:
                writer.writerow(
                    [
                        entry["timestamp"],
                        entry["threshold"],
                        detector,
                        result["label"],
                        result["confidence"],
                        result["ai_probability"],
                        entry["text"],
                    ]
                )
    return output.getvalue()


def load_example():
    st.session_state.text_to_analyze = SAMPLE_TEXT
    st.session_state.analysis_results = None


def clear_editor():
    st.session_state.text_to_analyze = ""
    st.session_state.analysis_results = None


if "text_to_analyze" not in st.session_state:
    st.session_state.text_to_analyze = ""
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">✳</span>signal<span style="color:#6f7a73"> / lab</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown('<div class="section-label">Analysis settings</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "AI classification threshold",
        min_value=0.1,
        max_value=0.9,
        value=float(Config.THRESHOLD),
        step=0.05,
        help="Higher thresholds require a stronger AI probability before a detector labels text AI-generated.",
    )
    st.caption(f"Minimum {Config.MIN_WORDS} words · Maximum {Config.MAX_CHARS:,} characters")
    st.markdown("---")
    st.markdown(
        '<div class="warning-note">This project uses a small demonstration model. '
        "Treat every result as a rough signal, never proof.</div>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Writing analysis · 01</div>
      <h1>Read between<br>the lines.</h1>
      <p class="subtle">Explore the patterns in a passage with two lightweight detectors.
      Your text stays in this session and is not saved by this app.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col, guide_col = st.columns([1.7, 0.8], gap="large")
with input_col:
    st.markdown('<div class="section-label">Your passage</div>', unsafe_allow_html=True)
    text = st.text_area(
        "Paste or write a passage to analyze",
        key="text_to_analyze",
        height=250,
        max_chars=Config.MAX_CHARS,
        placeholder="Paste at least 30 words here...",
        label_visibility="collapsed",
    )
    current_words = word_count(text)
    count_color = "#2f7459" if current_words >= Config.MIN_WORDS else "#a26d26"
    st.markdown(
        f'<div class="metric-strip"><strong>{current_words}</strong> words '
        f'<span style="padding:0 .45rem;color:#c5cec5">/</span> '
        f'<strong>{len(text):,}</strong> of {Config.MAX_CHARS:,} characters '
        f'<span style="float:right;color:{count_color}">'
        f'{"Ready to analyze" if current_words >= Config.MIN_WORDS else f"{Config.MIN_WORDS - current_words} more words needed"}</span></div>',
        unsafe_allow_html=True,
    )
    action_cols = st.columns([1.35, 1, 1, 2.4], vertical_alignment="center")
    with action_cols[0]:
        analyze_clicked = st.button("Analyze passage  →", type="primary", width="stretch")
    with action_cols[1]:
        st.button("Use example", width="stretch", on_click=load_example)
    with action_cols[2]:
        st.button("Clear", width="stretch", on_click=clear_editor)

with guide_col:
    st.markdown(
        """
        <div class="surface" style="margin-top:1.55rem">
          <div class="section-label">How to use</div>
          <div style="font:600 1rem 'Manrope',sans-serif;margin:.2rem 0 .45rem">A closer look, not a verdict.</div>
          <div class="result-caption" style="line-height:1.65">
            Add a passage of at least 30 words. Compare the rule-based writing
            signals with the machine-learning estimate, then use the details to
            understand what influenced each result.
          </div>
          <div style="margin-top:1rem"><span class="pill">Two independent signals</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if analyze_clicked:
    detectors = {
        "Rule-based": RuleBasedDetector(threshold=threshold),
        "Machine learning": MLTextDetector(threshold=threshold),
    }
    results = {}
    for name, detector in detectors.items():
        try:
            results[name] = detector.analyze(text)
        except FileNotFoundError as error:
            results[name] = {"error": str(error)}
    st.session_state.analysis_results = {
        "text": text,
        "threshold": threshold,
        "results": results,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if any("error" not in result for result in results.values()):
        st.session_state.analysis_history.insert(0, st.session_state.analysis_results)


st.markdown("")
analysis_tab, history_tab, about_tab = st.tabs(["Analysis", "Session history", "About"])

with analysis_tab:
    current = st.session_state.analysis_results
    if not current:
        st.markdown(
            '<div class="surface" style="text-align:center;padding:2rem 1rem">'
            '<div class="eyebrow">Your results will appear here</div>'
            '<div style="font:600 1.1rem Manrope,sans-serif;margin-top:.5rem">Ready when you are.</div>'
            '<div class="result-caption" style="margin-top:.35rem">Try the example passage or add your own text to get started.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        results = current["results"]
        successful = {name: result for name, result in results.items() if "error" not in result}
        if successful:
            ai_generated_votes = sum(result["label"] == "AI-Generated" for result in successful.values())
            overall = (
                "Detectors disagree"
                if ai_generated_votes not in (0, len(successful))
                else "AI-like patterns"
                if ai_generated_votes
                else "Human-like patterns"
            )
            st.markdown('<div class="section-label">Analysis summary</div>', unsafe_allow_html=True)
            summary_col, export_col = st.columns([2, 1], vertical_alignment="center")
            summary_style = (
                "pill--disagree"
                if overall == "Detectors disagree"
                else "pill--ai"
                if overall == "AI-like patterns"
                else ""
            )
            with summary_col:
                st.markdown(
                    f'<div class="surface"><span class="pill {summary_style}">{overall}</span>'
                    f'<div class="result-caption" style="margin-top:.75rem">Based on {len(successful)} detector{"s" if len(successful) != 1 else ""}. '
                    "This is an estimate, not proof.</div></div>",
                    unsafe_allow_html=True,
                )
            with export_col:
                report = json.dumps(current, indent=2, ensure_ascii=False)
                st.download_button(
                    "Download this report",
                    data=report,
                    file_name="signal-analysis.json",
                    mime="application/json",
                    width="stretch",
                )

            result_cols = st.columns(2, gap="large")
            for column, (name, result) in zip(result_cols, results.items()):
                with column:
                    st.markdown(f'<div class="section-label">{name}</div>', unsafe_allow_html=True)
                    if "error" in result:
                        st.error(result["error"])
                        continue
                    result_style = (
                        "result-card--disagree"
                        if overall == "Detectors disagree"
                        else "result-card--ai"
                        if result["label"] == "AI-Generated"
                        else ""
                    )
                    st.markdown(
                        f'<div class="surface result-card {result_style}"><div class="result-title">{result["label"]}</div>'
                        f'<div class="result-caption">{result["detector"]}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.metric("Classification confidence", f'{result["confidence"]}%')
                    st.progress(float(result["ai_probability"]), text="AI likelihood")
                    st.caption(f'AI likelihood: {result["ai_probability"] * 100:.1f}%')
                    if name == "Rule-based":
                        details = result["details"]
                        metric_cols = st.columns(2)
                        metric_cols[0].metric("Sentences", details["sentences"])
                        metric_cols[1].metric("Avg. sentence length", details["avg_sentence_length"])
                        metric_cols[0].metric("Burstiness", details["burstiness"])
                        metric_cols[1].metric("Vocabulary diversity", details["vocabulary_diversity"])
                    else:
                        st.caption(f'Model: {result["details"]["model"]}')
                    st.info(result["note"])

            st.markdown(
                '<div class="warning-note">AI-text detection is inherently uncertain. '
                "Short, edited, translated, or non-native writing can be misclassified. "
                "Do not use these estimates for high-stakes decisions.</div>",
                unsafe_allow_html=True,
            )
        else:
            for result in results.values():
                st.error(result["error"])

with history_tab:
    history = st.session_state.analysis_history
    st.markdown('<div class="section-label">This browser session only</div>', unsafe_allow_html=True)
    if history:
        history_action_cols = st.columns([1, 1, 4], vertical_alignment="center")
        with history_action_cols[0]:
            st.download_button(
                "Export history CSV",
                data=make_history_csv(history),
                file_name="signal-history.csv",
                mime="text/csv",
                width="stretch",
            )
        with history_action_cols[1]:
            if st.button("Clear history", width="stretch"):
                st.session_state.analysis_history = []
                st.rerun()
        for index, entry in enumerate(history, start=1):
            labels = " · ".join(
                f'{name}: {result.get("label", result.get("error"))}'
                for name, result in entry["results"].items()
            )
            with st.expander(f'#{index} · {labels} · {entry["timestamp"]}'):
                st.caption(
                    f'Threshold: {entry["threshold"]:.2f} · '
                    f'{word_count(entry["text"])} words · {len(entry["text"]):,} characters'
                )
                st.write(entry["text"])
                rows = result_rows(entry["results"])
                if rows:
                    st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.markdown(
            '<div class="surface"><div class="result-title">No analyses yet</div>'
            '<div class="result-caption">Your completed analyses will be listed here until this session ends.</div></div>',
            unsafe_allow_html=True,
        )

with about_tab:
    st.markdown('<div class="section-label">What is behind the estimate?</div>', unsafe_allow_html=True)
    about_col_a, about_col_b = st.columns(2, gap="large")
    with about_col_a:
        st.markdown(
            """
            <div class="surface">
              <div class="result-title">Rule-based detector</div>
              <div class="result-caption" style="line-height:1.7;margin-top:.5rem">
                Compares sentence-length variation and vocabulary diversity.
                These are simple stylistic signals and are not reliable proof of authorship.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with about_col_b:
        st.markdown(
            """
            <div class="surface">
              <div class="result-title">Machine-learning detector</div>
              <div class="result-caption" style="line-height:1.7;margin-top:.5rem">
                Uses the project's TF-IDF and logistic-regression model. Its included
                sample training set is small and intended for demonstration.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("")
    st.markdown(
        '<div class="warning-note">Results are estimates and can be wrong. '
        "They should not be used to judge a person, make academic or employment decisions, "
        "or establish who wrote a passage.</div>",
        unsafe_allow_html=True,
    )
