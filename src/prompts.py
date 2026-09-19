"""Prompt construction for the three policy modes (none / full / rag) and for the revise step."""
from __future__ import annotations

import json

from .policy_spec import COMPANY

SYSTEM_BASE = f"""You are a retention-offer drafting assistant for {COMPANY}, a telecom company. For one customer you choose the next retention action and draft the customer message. A person reviews everything before it is sent.

Return ONE JSON object with these fields:
- action: "OFFER" or "NO_ACTION".
- offer_id: LOYALTY_DISCOUNT (percentage discount on monthly charges), TERM_UPGRADE (discount for moving to a 12-month term), TECH_SUPPORT_TRIAL (Tech Support add-on at no charge for a few months), SERVICE_CHECKIN (non-promotional call invitation), or NONE when action is NO_ACTION.
- discount_pct and duration_months: the numbers in the offer terms (0 when the offer has none).
- channel: use "sms".
- message: the customer-facing text (empty string for NO_ACTION).
- rationale: one or two sentences for the human reviewer.
- citations: ids of the policy sections you relied on (empty list if no policy is provided)."""

POLICY_INSTRUCTIONS = {
    "none": "No policy documents are provided. Use good practice for compliant telecom marketing.",
    "full": "Follow the company policy below exactly. Cite the section ids (shown in square brackets) that you relied on.",
    "rag": ("Follow the policy sections below exactly; they were retrieved for this customer. Cite the section ids "
            "(shown in square brackets) that you relied on, and only ids shown below."),
}


def format_chunks(chunks) -> str:
    return "\n\n".join(f"[{c.id}]\n{c.text}" for c in chunks)


def system_prompt(policy_mode: str, policy_text: str) -> str:
    text = f"{SYSTEM_BASE}\n\n{POLICY_INSTRUCTIONS[policy_mode]}"
    return f"{text}\n\n{policy_text}" if policy_text else text


def _drivers(profile: dict) -> str:
    def fmt(raw, effect):
        return [f"{d['feature']} = {d['value']} ({effect})" for d in json.loads(raw or "[]")]
    parts = fmt(profile.get("drivers_json"), "raises risk")[:3] + fmt(profile.get("protective_json"), "lowers risk")[:1]
    return "; ".join(parts) or "none notable"


def customer_facts(profile: dict) -> str:
    return "\n".join([
        "Customer facts:",
        f"- contract: {profile['contract']}",
        f"- internet_service: {profile['internet_service']}",
        f"- tech_support: {profile['tech_support']}",
        f"- tenure_months: {profile['tenure_months']}",
        f"- monthly_charges: {float(profile['monthly_charges']):.2f}",
        f"- marketing_opt_in: {int(profile['marketing_opt_in'])}",
        f"- contacts_last_90d: {int(profile['contacts_last_90d'])}",
        f"- churn_risk_score: {float(profile['p_churn']):.2f} (0 to 1; never mention it to the customer)",
        f"- main factors behind the score: {_drivers(profile)}",
        "",
        "Choose the next retention action for this customer and return the JSON object."])


def revision_request(violations: list[str]) -> str:
    if not violations:
        return "Return a corrected JSON object."
    listed = "\n".join(f"- {v}" for v in violations)
    return (f"Your draft broke these policy rules:\n{listed}\n\n"
            "Return a corrected JSON object. Change only what is needed to fix the problems.")


def build_messages(policy_mode: str, policy_text: str, profile: dict,
                   previous_json: str | None = None, violations: list[str] | None = None) -> list[tuple[str, str]]:
    msgs = [("system", system_prompt(policy_mode, policy_text)), ("human", customer_facts(profile))]
    if violations is not None:
        msgs.append(("ai", previous_json or "(no valid JSON was returned)"))
        msgs.append(("human", revision_request(violations)))
    return msgs
