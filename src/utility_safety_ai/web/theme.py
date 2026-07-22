"""Commercial-style visual system for the Streamlit application."""

from __future__ import annotations

import html

import streamlit as st


def inject_theme() -> None:
    """Apply the product design tokens and responsive component styling."""
    st.markdown(
        """
        <style>
        :root {
          --bg:#05080f; --surface:#0a101c; --surface-2:#0f1726; --line:#1c2a3d;
          --text:#f5f7fb; --muted:#8d9bb0; --amber:#ffb000; --amber-2:#ff7a00;
          --cyan:#5ee7f2; --green:#38d996; --red:#ff5c70;
        }
        .stApp { color:var(--text); background:
          radial-gradient(900px 420px at 75% -10%,rgba(94,231,242,.07),transparent 62%),
          radial-gradient(700px 380px at 4% 18%,rgba(255,176,0,.055),transparent 64%),var(--bg); }
        .main .block-container { max-width:1480px;padding:1.1rem 2.1rem 4rem; }
        [data-testid="stSidebar"] { background:rgba(7,12,21,.96);border-right:1px solid var(--line); }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.65rem; }
        [data-testid="stSidebar"] hr { border-color:var(--line); }
        .usi-brand {display:flex;gap:.8rem;align-items:center;margin:.1rem 0 1.15rem}
        .usi-mark {width:40px;height:40px;border-radius:11px;display:grid;place-items:center;
          color:#05080f;font-weight:950;background:linear-gradient(145deg,var(--amber),var(--amber-2));
          box-shadow:0 12px 30px rgba(255,122,0,.2)}
        .usi-brand b{font-size:.9rem;letter-spacing:.01em}.usi-brand small{display:block;color:var(--muted);font-size:.67rem}
        .usi-hero {position:relative;overflow:hidden;border:1px solid var(--line);border-radius:22px;
          background:linear-gradient(130deg,rgba(15,23,38,.98),rgba(8,15,26,.94));padding:1.55rem 1.7rem 1.45rem;
          box-shadow:0 28px 80px -55px rgba(94,231,242,.6);margin-bottom:1rem}
        .usi-hero:after {content:"";position:absolute;width:440px;height:440px;right:-180px;top:-280px;
          border:1px solid rgba(94,231,242,.12);border-radius:50%;box-shadow:0 0 0 60px rgba(94,231,242,.02)}
        .usi-kicker {font-size:.67rem;font-weight:850;letter-spacing:.18em;color:var(--amber);text-transform:uppercase}
        .usi-hero h1{font-size:2rem;line-height:1.08;letter-spacing:-.045em;margin:.45rem 0;color:#fff}
        .usi-hero p{max-width:770px;color:#9eabc0;margin:0;font-size:.92rem}
        .usi-status {display:flex;gap:.45rem;align-items:center;margin-top:1rem;color:#b6c2d2;font-size:.72rem}
        .usi-dot {width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 5px rgba(56,217,150,.1)}
        .usi-flow {display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem;margin:.2rem 0 1rem}
        .usi-step {background:rgba(10,16,28,.82);border:1px solid var(--line);border-radius:12px;padding:.68rem .75rem;
          color:var(--muted);font-size:.7rem}.usi-step b{display:block;color:#dde5f0;font-size:.76rem;margin-top:.16rem}
        .usi-step span{color:var(--amber);font-weight:900;margin-right:.3rem}
        .usi-section {display:flex;align-items:end;justify-content:space-between;margin:.4rem 0 .75rem}
        .usi-section h2{font-size:1.1rem;margin:0;letter-spacing:-.02em}.usi-section p{font-size:.72rem;color:var(--muted);margin:0}
        .usi-boundary {border:1px solid rgba(255,176,0,.18);background:rgba(255,176,0,.045);border-radius:11px;
          padding:.7rem .85rem;color:#d4b979;font-size:.72rem;margin:.5rem 0}
        [data-testid="stMetric"] {background:linear-gradient(160deg,rgba(15,23,38,.96),rgba(9,15,25,.96));
          border:1px solid var(--line);border-radius:15px;padding:1rem 1rem .85rem;box-shadow:0 18px 45px -38px #000}
        [data-testid="stMetricLabel"] {color:var(--muted)} [data-testid="stMetricValue"]{font-weight:780;letter-spacing:-.035em}
        [data-baseweb="tab-list"] {gap:.32rem;padding:.28rem;background:rgba(10,16,28,.8);border:1px solid var(--line);border-radius:13px}
        [data-baseweb="tab"] {border-radius:9px;padding:.48rem .85rem;color:#a8b4c5}
        [data-baseweb="tab-highlight"] {background:linear-gradient(90deg,var(--amber),var(--amber-2));height:2px}
        [data-testid="stFileUploader"] {background:rgba(10,16,28,.72);border:1px dashed #2a3b53;border-radius:15px;padding:.35rem}
        [data-testid="stFileUploader"]:hover{border-color:rgba(94,231,242,.45)}
        .stButton>button,.stDownloadButton>button {border-radius:10px;min-height:2.65rem;font-weight:730;border-color:#2a3b53}
        .stButton>button[kind="primary"] {color:#080b10;border:0;background:linear-gradient(105deg,var(--amber),var(--amber-2));
          box-shadow:0 12px 28px -16px rgba(255,122,0,.85)}
        div[data-testid="stAlert"]{border-radius:12px;border-width:1px}
        [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
        .usi-empty {border:1px dashed #28374d;border-radius:16px;padding:2.2rem;text-align:center;color:var(--muted);background:rgba(9,15,25,.55)}
        .usi-empty b{display:block;color:#dce5f1;margin-bottom:.25rem}
        .usi-quality {display:grid;grid-template-columns:repeat(3,1fr);gap:.55rem;margin:.5rem 0}
        .usi-quality>div{border:1px solid var(--line);border-radius:12px;padding:.75rem;background:var(--surface)}
        .usi-quality strong{display:block;font-size:1.15rem}.usi-quality small{color:var(--muted)}
        @media(max-width:900px){.main .block-container{padding:1rem}.usi-flow{grid-template-columns:repeat(2,1fr)}.usi-quality{grid-template-columns:1fr}.usi-hero h1{font-size:1.55rem}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand() -> None:
    st.markdown(
        '<div class="usi-brand"><div class="usi-mark">U</div><div><b>UTILITY SAFETY</b><small>VISUAL INTELLIGENCE</small></div></div>',
        unsafe_allow_html=True,
    )


def hero(product: str, subtitle: str, ready: str, boundary: str) -> None:
    st.markdown(
        f"""
        <div class="usi-hero">
          <div class="usi-kicker">FIELD OPERATIONS · VISION AI</div>
          <h1>{html.escape(product)}</h1><p>{html.escape(subtitle)}</p>
          <div class="usi-status"><span class="usi-dot"></span>{html.escape(ready)}<span>·</span>{html.escape(boundary)}</div>
        </div>
        <div class="usi-flow">
          <div class="usi-step"><span>01</span>INPUT<b>Image / video / camera</b></div>
          <div class="usi-step"><span>02</span>POLICY<b>Zones & PPE rules</b></div>
          <div class="usi-step"><span>03</span>ANALYSE<b>Track & confirm</b></div>
          <div class="usi-step"><span>04</span>REVIEW<b>Results & reports</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    st.markdown(
        f'<div class="usi-section"><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p></div>',
        unsafe_allow_html=True,
    )
