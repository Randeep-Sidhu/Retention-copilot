"""HTML/CSS building blocks for the Streamlit app.

Palette: ivory, sand, espresso and a muted bronze. Serif headings, quiet cards, hairline borders.
Kept conservative (no CSS variables or grid) so the markup renders the same in every browser.
"""
from __future__ import annotations

import html

import streamlit as st

IVORY, SAND, LINE = "#FAF7F2", "#F2ECE3", "#E7DFD3"
ESPRESSO, TAUPE, MUTED = "#2A2420", "#7A6E63", "#8B8178"
BRONZE, GOLD = "#8C6A3F", "#B08D57"
SAGE, TERRACOTTA, OCHRE = "#6F8A6B", "#B4574A", "#B8893B"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&display=swap');
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
[data-testid="stSidebar"] { border-right: 1px solid #E7DFD3; }
[data-testid="stAppDeployButton"], .stDeployButton, footer { display: none !important; }
.rc-serif, .rc-word, .rc-hero h1, .rc-h, .rc-kpi .v, .rc-step .i { font-family: 'Cormorant Garamond', Georgia, 'Times New Roman', serif; }
.rc-word { font-size: 1.75rem; font-weight: 700; color: #2A2420; line-height: 1.1; margin: 2px 0 2px 0; }
.rc-word-sub { color: #7A6E63; font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 18px; }
.rc-hero { background: #2A2420; background: linear-gradient(135deg, #2A2420 0%, #43352B 100%); padding: 44px 48px 40px 48px;
  border-radius: 6px; margin-bottom: 24px; border-bottom: 3px solid #B08D57; }
.rc-eyebrow { color: #B08D57; font-size: 0.7rem; letter-spacing: 0.24em; text-transform: uppercase; margin-bottom: 14px; }
.rc-hero h1 { color: #F6EFE6; margin: 0 0 12px 0; padding: 0; font-size: 3.2rem; font-weight: 600; line-height: 1.04; letter-spacing: 0.005em; }
.rc-hero p { color: #D9CDBE; margin: 0; font-size: 1.08rem; line-height: 1.65; max-width: 780px; font-weight: 300; }
.rc-h { color: #2A2420; font-size: 2rem; font-weight: 600; margin: 2px 0 2px 0; line-height: 1.15; }
.rc-sub { color: #7A6E63; margin: 0 0 20px 0; font-size: 1rem; line-height: 1.5; }
.rc-label { color: #7A6E63; font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase; margin: 6px 0 8px 0; }
.rc-kpi { background: #FFFFFF; border: 1px solid #E7DFD3; border-radius: 6px; padding: 18px 20px 16px 20px; }
.rc-kpi .l { color: #7A6E63; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.17em; }
.rc-kpi .v { color: #2A2420; white-space: nowrap; font-size: 2.4rem; font-weight: 600; line-height: 1.1; margin: 8px 0 4px 0; }
.rc-kpi .n { color: #8B8178; font-size: 0.82rem; line-height: 1.35; }
.rc-step { background: #F2ECE3; border-top: 2px solid #B08D57; border-radius: 0 0 6px 6px; padding: 14px 16px 14px 16px; min-height: 158px; }
.rc-step .i { color: #B08D57; font-size: 1.7rem; font-weight: 600; line-height: 1; margin-bottom: 6px; }
.rc-step .t { font-weight: 600; color: #2A2420; margin-bottom: 4px; letter-spacing: 0.02em; }
.rc-step .d { color: #6B6157; font-size: 0.86rem; line-height: 1.45; }
.rc-card { background: #FFFFFF; border: 1px solid #E7DFD3; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px; }
.rc-card h4 { margin: 0 0 8px 0; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.17em; color: #7A6E63; font-weight: 600; }
.rc-kv { width: 100%; border-collapse: collapse; border: 0; }
.rc-kv td { padding: 4px 0; font-size: 0.92rem; color: #2A2420; vertical-align: top; border: 0; border-bottom: 1px solid #F0EAE0; }
.rc-kv tr:last-child td { border-bottom: 0; }
.rc-kv td.k { color: #7A6E63; width: 50%; }
.rc-pill { display: inline-block; padding: 2px 11px; border-radius: 999px; font-size: 0.74rem; font-weight: 600; margin: 2px 6px 2px 0; letter-spacing: 0.01em; }
.rc-pill.green { background: #E6EDE2; color: #4F6B4B; }
.rc-pill.red { background: #F5E3DF; color: #93453A; }
.rc-pill.amber { background: #F4E9D3; color: #7A5A22; }
.rc-pill.blue { background: #EFE5D6; color: #6B4F2A; }
.rc-pill.gray { background: #EFE9E0; color: #5F554B; }
.rc-phone { background: #F2ECE3; border-radius: 10px; padding: 16px; }
.rc-sms-wrap { text-align: right; }
.rc-sms { background: #3A312A; color: #F6EFE6; padding: 12px 16px; border-radius: 18px 18px 4px 18px; font-size: 0.95rem; line-height: 1.5;
  display: inline-block; max-width: 92%; text-align: left; }
.rc-sms.empty { background: #E2D9CB; color: #6B6157; font-style: italic; }
.rc-meta { color: #8B8178; font-size: 0.74rem; margin-top: 8px; text-align: right; }
.rc-trace { border: 1px solid #E7DFD3; border-left: 3px solid #B08D57; background: #FFFFFF; border-radius: 0 6px 6px 0; padding: 8px 14px; margin: 6px 0; }
.rc-trace.bad { border-left-color: #B4574A; }
.rc-trace.good { border-left-color: #6F8A6B; }
.rc-trace .n { font-weight: 600; color: #2A2420; letter-spacing: 0.01em; }
.rc-trace .t { float: right; color: #8B8178; font-family: 'SFMono-Regular', Consolas, monospace; font-size: 0.76rem; }
.rc-trace .d { color: #6B6157; font-size: 0.84rem; margin-top: 2px; word-wrap: break-word; }
.rc-risk { background: #E7DFD3; border-radius: 4px; height: 8px; margin: 8px 0 4px 0; }
.rc-risk div { height: 8px; border-radius: 4px; }
.rc-note { background: #F5EFE6; border-left: 3px solid #B08D57; border-radius: 0 6px 6px 0; padding: 11px 16px; color: #2A2420; font-size: 0.93rem; line-height: 1.5; margin: 6px 0 16px 0; }
.rc-foot { color: #8B8178; font-size: 0.78rem; margin-top: 32px; border-top: 1px solid #E7DFD3; padding-top: 12px; }
</style>
"""

STATUS = {
    "queued": ("Queued for review", "green"), "escalated": ("Escalated to a human", "red"),
    "no_action": ("No action", "gray"), "skipped_gate": ("Skipped by the value gate", "gray"),
    "queued_unverified": ("Queued, not verified", "amber"), "failed_schema": ("Invalid output", "red"),
    "error": ("Error", "red"),
}


def esc(x) -> str:
    return html.escape(str(x))


def plural(n, word: str) -> str:
    return f"{n} {word}" + ("" if int(n) == 1 else "s")


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def wordmark() -> None:
    st.markdown('<div class="rc-word">Retention Copilot</div><div class="rc-word-sub">Review app</div>',
                unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "") -> None:
    top = f'<div class="rc-eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    st.markdown(f'<div class="rc-hero">{top}<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def heading(title: str, sub: str = "") -> None:
    tail = f'<div class="rc-sub">{esc(sub)}</div>' if sub else ""
    st.markdown(f'<div class="rc-h">{esc(title)}</div>{tail}', unsafe_allow_html=True)


def label(text: str) -> None:
    st.markdown(f'<div class="rc-label">{esc(text)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="rc-note">{esc(text)}</div>', unsafe_allow_html=True)


def kpi(col, label_text: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="rc-kpi"><div class="l">{esc(label_text)}</div><div class="v">{esc(value)}</div>'
                 f'<div class="n">{esc(sub)}</div></div>', unsafe_allow_html=True)


def step(col, index: int, title: str, text: str) -> None:
    col.markdown(f'<div class="rc-step"><div class="i">{index:02d}</div><div class="t">{esc(title)}</div>'
                 f'<div class="d">{esc(text)}</div></div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "gray") -> str:
    return f'<span class="rc-pill {tone}">{esc(text)}</span>'


def status_pill(status: str) -> str:
    text, tone = STATUS.get(status, (status, "gray"))
    return pill(text, tone)


def card(title: str, rows: list[tuple[str, str]]) -> None:
    body = "".join(f'<tr><td class="k">{esc(k)}</td><td>{esc(v)}</td></tr>' for k, v in rows)
    st.markdown(f'<div class="rc-card"><h4>{esc(title)}</h4><table class="rc-kv">{body}</table></div>', unsafe_allow_html=True)


def risk_bar(p: float) -> None:
    colour = SAGE if p < 0.3 else OCHRE if p < 0.6 else TERRACOTTA
    text = "lower risk" if p < 0.3 else "medium risk" if p < 0.6 else "high risk"
    st.markdown(f'<div class="rc-card"><h4>Churn risk score</h4><div class="rc-serif" style="font-size:2rem;font-weight:600;color:#2A2420">{p:.2f}'
                f' <span style="font-family:inherit;font-size:0.85rem;color:#8B8178;font-weight:400">{text}</span></div>'
                f'<div class="rc-risk"><div style="width:{max(2, min(100, p * 100)):.0f}%;background:{colour}"></div></div></div>',
                unsafe_allow_html=True)


def sms(text: str, meta: str = "") -> None:
    body = esc(text).replace("\n", "<br>") if text else "no message (no action taken)"
    cls = "rc-sms" if text else "rc-sms empty"
    tail = f'<div class="rc-meta">{esc(meta)}</div>' if meta else ""
    st.markdown(f'<div class="rc-phone"><div class="rc-sms-wrap"><div class="{cls}">{body}</div></div>{tail}</div>',
                unsafe_allow_html=True)


def trace_step(node: str, t: float, detail: str, tone: str = "") -> None:
    st.markdown(f'<div class="rc-trace {tone}"><span class="t">{t:.2f}s</span><span class="n">{esc(node)}</span>'
                f'<div class="d">{esc(detail)}</div></div>', unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f'<div class="rc-foot">{esc(text)}</div>', unsafe_allow_html=True)
