"""A restrained inspection-workspace theme for the Streamlit application."""

from __future__ import annotations

import html

import streamlit as st


def inject_theme() -> None:
    """Apply readable type, quiet surfaces and consistent workspace components."""
    st.markdown(
        """
        <style>
        :root {
          --bg:#f7f8fa; --surface:#fff; --surface-2:#edf1f5; --line:#dce2e9;
          --text:#172a3a; --muted:#5f7181; --amber:#e9a23b; --navy:#16354a;
          --green:#247568; --red:#b6483e;
        }
        .stApp {color:var(--text);background:var(--bg)}
        [data-testid="stMainBlockContainer"]{max-width:1500px;padding:1.5rem 2.5rem 3rem}
        [data-testid="stHeader"]{background:transparent}
        [data-testid="stSidebar"]{background:#eef2f6;border-right:1px solid var(--line)}
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.6rem}
        [data-testid="stSidebar"] h3{font-size:.9rem;font-weight:650;margin-top:.5rem}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"]{font-size:.8rem;line-height:1.5}
        p,li{line-height:1.6}
        h1,h2,h3{font-family:inherit;color:var(--text);letter-spacing:-.025em}
        .usi-brand{display:flex;gap:.7rem;align-items:center;margin:.1rem 0 1.1rem}
        .usi-mark{width:39px;height:42px;display:grid;place-items:center;background:var(--navy);
          color:#fff;border-bottom:4px solid var(--amber);font-size:.9rem;font-weight:750;letter-spacing:-.06em}
        .usi-brand b{font-size:.95rem;letter-spacing:.08em;line-height:1.25;display:block}
        .usi-brand small{display:block;color:var(--muted);font-size:.75rem;margin-top:.2rem}
        .usi-hero{display:flex;justify-content:space-between;align-items:center;gap:1rem;
          padding:.35rem 0 1.3rem;border-bottom:1px solid var(--line);margin-bottom:1.25rem}
        .usi-kicker{font-size:.75rem;line-height:1.5;letter-spacing:.14em;font-weight:650;color:var(--muted)}
        .usi-hero h1{font-size:1.9rem;line-height:1.2;margin:.25rem 0 .3rem;font-weight:680}
        .usi-hero p{color:var(--muted);margin:0;font-size:1rem}
        .usi-status{text-align:right;color:var(--muted);font-size:.8rem;line-height:1.75;white-space:nowrap}
        .usi-status b{display:block;color:var(--navy);font-weight:600}
        .usi-section{display:flex;gap:1rem;align-items:baseline;justify-content:space-between;margin:.5rem 0 .7rem}
        .usi-section h2{font-size:1.15rem;margin:0;font-weight:650}
        .usi-section p{font-size:.875rem;color:var(--muted);margin:0}
        .usi-boundary{padding:.6rem 0;color:var(--muted);font-size:.8rem;border-top:1px solid var(--line);margin-top:.6rem}
        [data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:.9rem 1rem}
        [data-testid="stMetricLabel"]{color:var(--muted);font-size:.85rem}
        [data-testid="stMetricValue"]{font-size:1.8rem;font-weight:650;color:var(--navy);letter-spacing:-.04em}
        [data-testid="stSegmentedControl"]{margin-bottom:.5rem}
        [data-baseweb="tab-list"]{gap:1.3rem;border-bottom:1px solid var(--line)}
        [data-baseweb="tab"]{font-size:.9rem;padding:.55rem .1rem;color:var(--muted)}
        [data-baseweb="tab-highlight"]{background:var(--navy);height:2px}
        [data-testid="stFileUploader"]{background:var(--surface);border:1px dashed #b4c2cf;border-radius:9px;padding:.4rem}
        [data-testid="stFileUploader"]:hover{border-color:var(--navy)}
        .stButton>button,.stDownloadButton>button{border-radius:7px;min-height:2.7rem;font-weight:600;border-color:#cad3dd}
        .stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{color:#fff;border:1px solid var(--navy);background:var(--navy)}
        .stButton>button[kind="primary"]:hover{background:#234d66;border-color:#234d66}
        div[data-testid="stAlert"]{border-radius:8px;border-width:1px}
        [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:8px;overflow:hidden}
        [data-testid="stImage"] img{border-radius:9px;max-height:560px;object-fit:contain;background:var(--surface-2)}
        .usi-empty{border:1px dashed #bdcbd7;border-radius:9px;padding:2rem;color:var(--muted);background:var(--surface)}
        .usi-empty b{display:block;color:var(--text);font-size:1.1rem;margin-bottom:.35rem}
        .usi-quality{display:grid;grid-template-columns:repeat(3,1fr);gap:1.5rem;margin:.7rem 0 1rem;padding:1rem 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
        .usi-quality strong{display:block;font-size:1.4rem;font-weight:650;color:var(--navy)}
        .usi-quality small{color:var(--muted);font-size:.85rem}
        .usi-scene-note{border-left:3px solid var(--amber);padding:.1rem 0 .1rem .8rem;margin:1rem 0;color:var(--muted);font-size:.9rem}
        .usi-run-heading{display:flex;gap:1rem;align-items:center;justify-content:space-between;margin:.5rem 0 1rem}
        .usi-run-heading h2{font-size:1.3rem;margin:0;font-weight:650}
        .usi-run-heading p{font-size:.8rem;color:var(--muted);margin:.2rem 0 0}
        .usi-summary-row{padding:.7rem 0;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:1rem;font-size:.9rem}
        .usi-summary-row span{color:var(--muted)}
        @media(max-width:900px){
          [data-testid="stMainBlockContainer"]{padding:1rem 1.1rem 2rem}
          .usi-hero h1{font-size:1.5rem}.usi-hero{align-items:flex-start}.usi-status{display:none}
          .usi-section{display:block}.usi-section p{margin-top:.25rem}
          .usi-quality{gap:.8rem}.usi-quality small{font-size:.8rem}
        }
        @media(prefers-reduced-motion:reduce){*{animation-duration:.01ms!important;transition-duration:.01ms!important}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand() -> None:
    st.markdown(
        '<div class="usi-brand"><div class="usi-mark">US</div>'
        "<div><b>UTILITY SAFETY</b><small>Inspection workspace</small></div></div>",
        unsafe_allow_html=True,
    )


def hero(product: str, subtitle: str, ready: str, boundary: str) -> None:
    st.markdown(
        f'<div class="usi-hero"><div><div class="usi-kicker">SITE OPERATIONS</div>'
        f"<h1>{html.escape(product)}</h1><p>{html.escape(subtitle)}</p></div>"
        f'<div class="usi-status"><b>{html.escape(ready)}</b>{html.escape(boundary)}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    st.markdown(
        f'<div class="usi-section"><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p></div>',
        unsafe_allow_html=True,
    )
