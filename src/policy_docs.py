"""Render the policy documents (policies/*.md) from policy_spec.py.  Run: python -m src.policy_docs"""
from __future__ import annotations

from pathlib import Path

from . import policy_spec as S

POLICY_DIR = Path(__file__).resolve().parents[1] / "policies"
NEW = S.NEW_CUSTOMER_MAX_TENURE
LOY, TERM, TRIAL, CHECK = (S.OFFERS[k] for k in ("LOYALTY_DISCOUNT", "TERM_UPGRADE", "TECH_SUPPORT_TRIAL", "SERVICE_CHECKIN"))
NICE = {"TECH_SUPPORT_TRIAL": "Tech support trial", "TERM_UPGRADE": "Term upgrade",
        "LOYALTY_DISCOUNT": "Loyalty discount", "SERVICE_CHECKIN": "Service check-in"}


def offer_catalog() -> str:
    return f"""# Offer catalog

{S.COMPANY} uses four retention actions. Every drafted action is reviewed by a person before anything is sent.

## Loyalty discount
offer_id: LOYALTY_DISCOUNT. A percentage discount on monthly charges for a limited number of months.
- Eligible: month-to-month or one-year contract, at least {LOY['min_tenure']} months of tenure, marketing consent on file, and fewer than {S.MAX_CONTACTS_90D} outreach contacts in the last 90 days.
- Standard terms: {S.STANDARD_DISCOUNT_PCT}% off for up to {S.MAX_DISCOUNT_MONTHS} months. Limits by contract and risk tier are in the discount caps document.
- Use it when no service-fit offer applies; the segment playbooks give the preferred order.
- Not available on two-year contracts or to customers in their first {NEW} months.

## Term upgrade
offer_id: TERM_UPGRADE. A discount in exchange for moving from a month-to-month contract to a 12-month term.
- Eligible: month-to-month contract, at least {TERM['min_tenure']} months of tenure, marketing consent on file.
- Fixed terms: {TERM['fixed_pct']}% off for {TERM['fixed_months']} months (discount_pct {TERM['fixed_pct']}, duration_months {TERM['fixed_months']}). The message must say the offer applies when the customer moves to a 12-month term.
- Not available on one-year or two-year contracts.

## Tech support trial
offer_id: TECH_SUPPORT_TRIAL. The Tech Support add-on at no charge for {TRIAL['fixed_months']} months.
- Eligible: the customer has an internet service (DSL or Fiber optic), does not currently have Tech Support, and has given marketing consent.
- Fixed terms: discount_pct {TRIAL['fixed_pct']}, duration_months {TRIAL['fixed_months']}. The word "free" may be used for this offer only.
- Not available to phone-only customers or to customers who already have Tech Support.

## Service check-in
offer_id: SERVICE_CHECKIN. A friendly, non-promotional invitation to talk to a service adviser about the customer's plan.
- Eligible: any customer while the contact limit has not been reached. It does not need marketing consent because it is a service message.
- Fixed terms: discount_pct 0, duration_months 0. The message contains no numbers and no promotional language.
- Use it when no other offer is eligible, or when the segment playbook lists it as the best fit.
"""


def discount_caps() -> str:
    caps = S.MAX_DISCOUNT_PCT
    return f"""# Discount caps

Limits that apply to every percentage discount. A drafted discount above these limits must be rejected.

## Month-to-month contracts
Loyalty discounts may not exceed {caps['Month-to-month']}% and may last at most {S.MAX_DISCOUNT_MONTHS} months. The standard maximum is {S.STANDARD_DISCOUNT_PCT}%; {caps['Month-to-month']}% is allowed only for high-risk customers (see risk tiers).

## One-year contracts
Loyalty discounts may not exceed {caps['One year']}% and may last at most {S.MAX_DISCOUNT_MONTHS} months. The high-risk tier does not raise this cap.

## Two-year contracts
No percentage discounts of any kind. Only the tech support trial (if eligible) or a service check-in may be offered.

## New customers 0-{NEW} months
No discounts of any kind for customers with {NEW} months of tenure or less. Offer the tech support trial (if eligible) or a service check-in.

## Risk tiers
The scoring model provides a churn risk score between 0 and 1. A score of {S.HIGH_RISK_THRESHOLD:.2f} or higher is the high-risk tier; anything lower is the standard tier. The standard tier allows at most {S.STANDARD_DISCOUNT_PCT}%. Only month-to-month customers in the high-risk tier may receive up to {caps['Month-to-month']}%.

## Absolute monthly cap
A discount may never reduce the bill by more than ${S.ABS_DISCOUNT_CAP_PER_MONTH:.0f} per month. The largest allowed percentage is {int(S.ABS_DISCOUNT_CAP_PER_MONTH * 100)} divided by the monthly charge, rounded down. For a $120 monthly charge the largest allowed discount is 12%.
"""


def consent_and_contact() -> str:
    return f"""# Consent and contact rules

## Marketing consent
Promotional messages (any offer with a price reduction or a trial) may only be sent to customers with marketing_opt_in = 1. If marketing_opt_in = 0, the only permitted action is the service check-in.

## Contact frequency
A customer may receive at most {S.MAX_CONTACTS_90D} outreach contacts in any 90 days. If contacts_last_90d is {S.MAX_CONTACTS_90D} or more, take no action: the action must be NO_ACTION with offer_id NONE and an empty message.

## Required opt-out line
Every promotional message must contain this exact line: "{S.OPT_OUT_LINE}".

## Channel limits
SMS messages may be at most {S.SMS_MAX_CHARS} characters. Email messages may be at most {S.EMAIL_MAX_CHARS} characters. Count every character, including spaces and the opt-out line.

## Service messages
The service check-in is a service message, not a promotion. It may be sent without marketing consent, it does not need the opt-out line, and it must not use promotional words such as discount, offer, save, deal or free.
"""


def message_standards() -> str:
    bad = ", ".join(f'"{p}"' for p in S.PROHIBITED_PHRASES_FOR_DOCS)
    return f"""# Message standards

## Structure
Start with "Hello," (never a name). One sentence of thanks or value, then the exact offer terms, then one clear next step (for example "Reply YES to accept"), then the opt-out line for promotional messages. Keep the tone warm, plain and calm.

## Numbers must match the offer
The only numbers allowed in a message are the offer terms. State the discount percentage and the number of months exactly as in the offer (for example 10 and 6). Term upgrade messages state 10 and 12; tech support trial messages state 3. Service check-in messages contain no numbers at all.

## Prohibited language
Never use pressure or absolute promises: {bad}. Never use the word "free" except for the tech support trial.

## Protected characteristics
Never mention or hint at a customer's age, gender, marital or family status, and never use these to choose an offer. Words such as senior, elderly, family, children, married or spouse must not appear.

## Predictions and risk scores
Never tell a customer that we expect them to leave. Do not mention scores, predictions, cancelling, churn, competitors or switching providers. The message should read as a thank-you, not as a retention attempt.

## Examples of acceptable messages
- Loyalty discount: "Hello, thank you for being with {S.COMPANY}. We would like to offer you 10% off your monthly charges for 6 months. Reply YES to accept. {S.OPT_OUT_LINE}."
- Term upgrade: "Hello, thank you for staying with {S.COMPANY}. Move to a 12-month term and enjoy 10% off your monthly charges for 12 months. Reply YES to accept. {S.OPT_OUT_LINE}."
- Tech support trial: "Hello, we would like to add Tech Support to your plan at no charge for 3 months. Reply YES to accept. {S.OPT_OUT_LINE}."
- Service check-in: "Hello, we would love to make sure your plan still fits your needs. A {S.COMPANY} adviser can call you at a time that suits you. Reply CALL and we will arrange it."
"""


def _ladder_reason(contract: str, internet: str) -> str:
    if contract == "Two year":
        return "Customers on a two-year contract are already committed, so discounts are not used."
    if internet == "Fiber optic":
        return "Fiber customers most often need reliability and support, so service-fit offers come before price offers."
    if internet == "DSL":
        return "DSL customers respond well to a longer commitment with support included before a plain discount."
    return "Phone-only customers have no add-ons to offer, so commitment and price offers come first."


def segment_playbooks() -> str:
    parts = ["# Segment playbooks\n",
             "Preferred order of offers by contract and internet service. Choose the first offer in the list that the "
             "customer is eligible for; skip any offer whose eligibility rule fails. Company guidance, not a statistical finding.\n"]
    for (contract, internet), ladder in S.LADDER.items():
        label = "no internet (phone only)" if internet == "No" else internet
        steps = "; ".join(f"{i}. {NICE[o]} ({o})" for i, o in enumerate(ladder, 1))
        parts.append(f"## {contract}, {label}\nPreferred order: {steps}.\n{_ladder_reason(contract, internet)}\n")
    return "\n".join(parts)


def escalation() -> str:
    return f"""# Escalation and review

## Human approval
Every drafted action goes to a review queue. Nothing is sent automatically. The reviewer sees the drafted message, the cited policy sections and any review flags.

## High-value accounts
Monthly charges above ${S.HIGH_VALUE_MONTHLY:.0f} are high-value accounts. Any discount offer for them is flagged for priority review, and the absolute monthly cap in the discount caps document still applies.

## New customers
Customers with {NEW} months of tenure or less are new customers. They are flagged for review and may only receive the tech support trial (if eligible) or a service check-in.

## When to take no action
Take no action (NO_ACTION, offer_id NONE, empty message) only when the contact limit is reached. In every other case at least a service check-in is available, so an offer must be chosen.
"""


DOCS = {"offer_catalog.md": offer_catalog, "discount_caps.md": discount_caps,
        "consent_and_contact.md": consent_and_contact, "message_standards.md": message_standards,
        "segment_playbooks.md": segment_playbooks, "escalation.md": escalation}


def main():
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in DOCS.items():
        (POLICY_DIR / name).write_text(fn().rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {len(DOCS)} policy documents to {POLICY_DIR}")


if __name__ == "__main__":
    main()
