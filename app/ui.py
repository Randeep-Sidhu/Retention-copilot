"""Small HTML/CSS building blocks for the Streamlit app.

Kept deliberately conservative (no CSS variables or grid) so the markup renders the same in every browser.
"""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.rc-hero { background: #0f62fe; background: linear-gradient(120deg, #0a1f5c 0%, #0f62fe 100%); color: #ffffff;
  padding: 30px 34px; border-radius: 14px; margin-bottom: 18px; }
.rc-hero h1 { color: #ffffff; margin: 0 0 8px 0; padding: 0; font-size: 2.2rem; font-weight: 600; }
.rc-hero p { color: #dbe7ff; margin: 0; font-size: 1.05rem; line-height: 1.5; max-width: 900px; }
.rc-tag { display: inline-block; background: rgba(255,255,255,0.18); color: #ffffff; padding: 3px 11px;
  border-radius: 999px; font-size: 0.78rem; margin: 16px 8px 0 0; }
.rc-kpi { background: #ffffff; border: 1px solid #e0e0e0; border-top: 3px solid #0f62fe; border-radius: 10px; padding: 16px 18px; }
.rc-kpi .l { color: #525252; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
.rc-kpi .v { color: #161616; font-size: 2rem; font-weight: 600; line-height: 1.15; margin: 4px 0 2px 0; }
.rc-kpi .n { color: #6f6f6f; font-size: 0.82rem; }
.rc-step { background: #f4f4f4; border-radius: 10px; padding: 14px 14px 12px 14px; min-height: 138px; }
.rc-step .i { display: inline-block; background: #0f62fe; color: #ffffff; width: 24px; height: 24px; line-height: 24px;
  text-align: center; border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-bottom: 8px; }
.rc-step .t { font-weight: 600; color: #161616; margin-bottom: 4px; }
.rc-step .d { color: #525252; font-size: 0.85rem; line-height: 1.4; }
.rc-card { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
.rc-card h4 { margin: 0 0 8px 0; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em; color: #525252; font-weight: 600; }
.rc-kv { width: 100%; border-collapse: collapse; }
.rc-kv td { padding: 3px 0; font-size: 0.9rem; color: #161616; vertical-align: top; }
.rc-kv td.k { color: #6f6f6f; width: 46%; }
.rc-pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.76rem; font-weight: 600; margin: 2px 6px 2px 0; }
.rc-pill.green { background: #defbe6; color: #0e6027; }
.rc-pill.red { background: #fff1f1; color: #a2191f; }
.rc-pill.amber { background: #fcf4d6; color: #684e00; }
.rc-pill.blue { background: #edf5ff; color: #0043ce; }
.rc-pill.gray { background: #f4f4f4; color: #393939; }
.rc-phone { background: #f4f4f4; border-radius: 16px; padding: 14px; }
.rc-sms-wrap { text-align: right; }
.rc-sms { background: #0f62fe; color: #ffffff; padding: 11px 15px; border-radius: 18px; font-size: 0.95rem; line-height: 1.45;
  display: inline-block; max-width: 92%; text-align: left; }
.rc-sms.empty { background: #e0e0e0; color: #525252; font-style: italic; }
.rc-meta { color: #6f6f6f; font-size: 0.74rem; margin-top: 6px; text-align: right; }
.rc-trace { border: 1px solid #e0e0e0; border-left: 3px solid #0f62fe; background: #ffffff; border-radius: 0 8px 8px 0;
  padding: 8px 14px; margin: 6px 0; }
.rc-trace.bad { border-left-color: #da1e28; }
.rc-trace.good { border-left-color: #24a148; }
.rc-trace .n { font-weight: 600; color: #161616; }
.rc-trace .t { float: right; color: #6f6f6f; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }
.rc-trace .d { color: #525252; font-size: 0.84rem; margin-top: 2px; word-wrap: break-word; }
.rc-risk { background: #e0e0e0; border-radius: 6px; height: 10px; margin: 6px 0 4px 0; }
.rc-risk div { height: 10px; border-radius: 6px; }
.rc-note { background: #edf5ff; border-left: 3px solid #0f62fe; border-radius: 0 8px 8px 0; padding: 10px 14px; color: #161616; font-size: 0.92rem; margin: 8px 0 14px 0; }
.rc-sub { color: #525252; margin: -6px 0 14px 0; font-size: 0.95rem; }
.rc-foot { color: #6f6f6f; font-size: 0.8rem; margin-top: 28px; border-top: 1px solid #e0e0e0; padding-top: 10px; }
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


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, tags=()) -> None:
    chips = "".join(f'<span class="rc-tag">{esc(t)}</span>' for t in tags)
    st.markdown(f'<div class="rc-hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p>{chips}</div>', unsafe_allow_html=True)


def heading(title: str, sub: str = "") -> None:
    st.markdown(f"### {title}")
    if sub:
        st.markdown(f'<div class="rc-sub">{esc(sub)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="rc-note">{esc(text)}</div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="rc-kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
                 f'<div class="n">{esc(sub)}</div></div>', unsafe_allow_html=True)


def step(col, index: int, title: str, text: str) -> None:
    col.markdown(f'<div class="rc-step"><div class="i">{index}</div><div class="t">{esc(title)}</div>'
                 f'<div class="d">{esc(text)}</div></div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "gray") -> str:
    return f'<span class="rc-pill {tone}">{esc(text)}</span>'


def status_pill(status: str) -> str:
    label, tone = STATUS.get(status, (status, "gray"))
    return pill(label, tone)


def card(title: str, rows: list[tuple[str, str]]) -> None:
    body = "".join(f'<tr><td class="k">{esc(k)}</td><td>{esc(v)}</td></tr>' for k, v in rows)
    st.markdown(f'<div class="rc-card"><h4>{esc(title)}</h4><table class="rc-kv">{body}</table></div>', unsafe_allow_html=True)


def risk_bar(p: float) -> None:
    colour = "#24a148" if p < 0.3 else "#f1c21b" if p < 0.6 else "#da1e28"
    label = "lower risk" if p < 0.3 else "medium risk" if p < 0.6 else "high risk"
    st.markdown(f'<div class="rc-card"><h4>Churn risk score</h4><div style="font-size:1.6rem;font-weight:600">{p:.2f}'
                f' <span style="font-size:0.85rem;color:#6f6f6f;font-weight:400">{label}</span></div>'
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
