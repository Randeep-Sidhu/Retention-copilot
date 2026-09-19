"""Prompt construction for the three policy modes (none / full / rag) and for the revise step."""
from __future__ import annotations

import json
import re

from .policy_spec import COMPANY, MAX_CONTACTS_90D

SYSTEM_BASE = f"""You are a retention-offer drafting assistant for {COMPANY}, a telecom company. For one customer you choose the next retention action and draft the customer message. A person reviews everything before it is sent.

Return ONE JSON object with these fields, in this order:
- rationale: think first, in one or two sentences: the customer's contract and internet service, which offer fits, and why.
- action: "OFFER" or "NO_ACTION".
- offer_id: LOYALTY_DISCOUNT (percentage discount on monthly charges), TERM_UPGRADE (discount for moving to a 12-month term), TECH_SUPPORT_TRIAL (Tech Support add-on at no charge for a few months), SERVICE_CHECKIN (non-promotional call invitation), or NONE when action is NO_ACTION.
- discount_pct and duration_months: the numbers in the offer terms (0 when the offer has none).
- channel: use "sms".
- message: the text sent to the customer, a short SMS of one or two sentences with no sign-off (empty string for NO_ACTION). The customer facts are for your decision only; do not repeat them in the message.
- citations: ids of the policy sections you relied on (empty list if no policy is provided)."""

_PROCEDURE = f"""
How to decide (work through these in order):
1. Contact limit: if contacts_last_90d is {MAX_CONTACTS_90D} or more, the action is NO_ACTION with offer_id NONE, discount_pct 0, duration_months 0 and an empty message.
2. Consent: if marketing_opt_in is 0, the only allowed offer is SERVICE_CHECKIN.
3. Otherwise open the playbook for the customer's contract and internet service and go down its preferred order from the top. Take the FIRST offer whose eligibility rules in the catalog all hold for this customer's tenure, contract, internet_service and tech_support. Skip any offer that fails a rule.
4. Use that offer's exact terms and stay within the discount caps.
5. Write the message following the message standards."""

POLICY_INSTRUCTIONS = {
    "none": "No policy documents are provided. Use good practice for compliant telecom marketing.",
    "full": ("Follow the company policy below exactly. Cite the section ids (shown in square brackets) that you relied on."
             + _PROCEDURE),
    "rag": ("Follow the policy sections below exactly; they were retrieved for this customer. Cite the section ids "
            "(shown in square brackets) that you relied on, and only ids shown below." + _PROCEDURE),
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


_MESSAGE_RULES = {"length", "numbers", "forbidden_language", "protected_terms", "prediction_terms", "opt_out_line",
                  "promo_in_service_message"}
_REWRITE_HINT = ("Rewrite the whole message from scratch instead of editing it, as a short SMS: 'Hello,' + one sentence "
                 "that states only the offer terms + the next step (for example 'Reply YES to accept.') + the opt-out "
                 "line if the offer is promotional. No sign-off, no facts about the customer, at most 250 characters.")


def revision_request(violations: list[str]) -> str:
    if not violations:
        return "Return a corrected JSON object."
    rules = {m.group(1) for v in violations if (m := re.match(r"\[(\w+)\]", v))}
    listed = "\n".join(f"- {v}" for v in violations)
    tail = _REWRITE_HINT if rules & _MESSAGE_RULES else "Change only what is needed to fix the problems."
    return f"Your draft broke these policy rules:\n{listed}\n\nReturn a corrected JSON object. {tail}"


def build_messages(policy_mode: str, policy_text: str, profile: dict,
                   previous_json: str | None = None, violations: list[str] | None = None) -> list[tuple[str, str]]:
    msgs = [("system", system_prompt(policy_mode, policy_text)), ("human", customer_facts(profile))]
    if violations is not None:
        msgs.append(("ai", previous_json or "(no valid JSON was returned)"))
        msgs.append(("human", revision_request(violations)))
    return msgs
