"""Deterministic policy engine: eligibility, terms, preferred offer and the verifier used by the agent.

`check()` returns actionable violations that are fed back to the LLM for revision. It never calls a model.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .policy_spec import (ABS_DISCOUNT_CAP_PER_MONTH, EMAIL_MAX_CHARS, FORBIDDEN_MARKETING, HIGH_RISK_THRESHOLD,
                          HIGH_VALUE_MONTHLY, LADDER, MAX_CONTACTS_90D, MAX_DISCOUNT_MONTHS, MAX_DISCOUNT_PCT,
                          NEW_CUSTOMER_MAX_TENURE, OFFERS, OPT_OUT_LINE, PREDICTION_TERMS, PROMO_WORDS,
                          PROTECTED_TERMS, SMS_MAX_CHARS, STANDARD_DISCOUNT_PCT, offer_section_id)
from .schemas import Proposal


@dataclass(frozen=True)
class Violation:
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.detail}"


# ----------------------------------------------------------------------------- eligibility and terms
def contact_limit_reached(profile: dict) -> bool:
    return int(profile["contacts_last_90d"]) >= MAX_CONTACTS_90D


def ineligibility(offer_id: str, profile: dict) -> tuple[str, str] | None:
    """Why this offer cannot be sent to this customer: (rule, explanation), or None if eligible."""
    o = OFFERS[offer_id]
    if o["promotional"] and not int(profile["marketing_opt_in"]):
        return "consent", (f"customer has not opted in to marketing, so promotional offer {offer_id} is not allowed "
                           "(only SERVICE_CHECKIN is)")
    if profile["contract"] not in o["contracts"]:
        return "eligibility", f"{offer_id} is not available on a {profile['contract']} contract"
    if int(profile["tenure_months"]) < o["min_tenure"]:
        return "eligibility", (f"{offer_id} needs at least {o['min_tenure']} months of tenure "
                               f"(customer has {profile['tenure_months']})")
    if o["needs_internet"] and profile["internet_service"] == "No":
        return "eligibility", f"{offer_id} requires an internet service; this customer is phone-only"
    if o["needs_no_tech_support"] and profile["tech_support"] != "No":
        return "eligibility", f"{offer_id} is only for internet customers without tech support"
    return None


def eligible_offers(profile: dict) -> list[str]:
    if contact_limit_reached(profile):
        return []
    return [oid for oid in OFFERS if ineligibility(oid, profile) is None]


def max_loyalty_pct(profile: dict) -> int:
    """Highest allowed loyalty-discount percentage: contract cap, risk tier and the $/month absolute cap."""
    contract_cap = MAX_DISCOUNT_PCT[profile["contract"]]
    high_risk = profile["contract"] == "Month-to-month" and float(profile["p_churn"]) >= HIGH_RISK_THRESHOLD
    tier_cap = MAX_DISCOUNT_PCT["Month-to-month"] if high_risk else STANDARD_DISCOUNT_PCT
    abs_cap = math.floor(ABS_DISCOUNT_CAP_PER_MONTH * 100 / float(profile["monthly_charges"]))
    return max(0, min(contract_cap, tier_cap, abs_cap))


def preferred_offer(profile: dict) -> str:
    """First eligible entry of the segment's ladder, or NONE when the contact limit blocks outreach."""
    if contact_limit_reached(profile):
        return "NONE"
    for oid in LADDER[(profile["contract"], profile["internet_service"])]:
        if ineligibility(oid, profile) is None:
            return oid
    return "SERVICE_CHECKIN"


def review_flags(profile: dict, proposal: Proposal) -> list[str]:
    flags = []
    if proposal.action == "OFFER" and proposal.discount_pct > 0 and float(profile["monthly_charges"]) > HIGH_VALUE_MONTHLY:
        flags.append("high_value_account")
    if int(profile["tenure_months"]) <= NEW_CUSTOMER_MAX_TENURE:
        flags.append("new_customer")
    return flags


# ----------------------------------------------------------------------------- verifier
def _snippet(text: str, pattern: str, width: int = 26) -> str:
    """Short quote of the offending text, so a small model can see exactly what to remove."""
    m = re.search(pattern, text, flags=re.I)
    if not m:
        return ""
    lo, hi = max(0, m.start() - width), min(len(text), m.end() + width)
    return "..." + " ".join(text[lo:hi].split()) + "..."


def _hits(patterns, text):
    return sorted({m.group(0).lower() for p in patterns for m in re.finditer(p, text, flags=re.I)})


def _citation_violations(p: Proposal, profile: dict, valid_ids) -> list[Violation]:
    v = []
    if not p.citations:
        v.append(Violation("citations", "cite at least one policy section id (for example offer_catalog#loyalty-discount)"))
    if valid_ids is not None:
        bad = [c for c in p.citations if c not in valid_ids]
        if bad:
            v.append(Violation("citations", f"these section ids do not exist: {bad}; cite only ids that were retrieved"))
    if p.action == "OFFER" and p.offer_id in OFFERS:
        need = offer_section_id(p.offer_id)
        if need not in p.citations:
            v.append(Violation("citations", f"cite the catalog section for the chosen offer: {need}"))
    if p.action == "NO_ACTION" and contact_limit_reached(profile):
        if "consent_and_contact#contact-frequency" not in p.citations:
            v.append(Violation("citations", "cite consent_and_contact#contact-frequency to justify NO_ACTION"))
    return v


def _terms_violations(p: Proposal, profile: dict) -> list[Violation]:
    o, v = OFFERS[p.offer_id], []
    if p.offer_id == "LOYALTY_DISCOUNT":
        cap = max_loyalty_pct(profile)
        if not 1 <= p.discount_pct <= cap:
            v.append(Violation("discount_cap", (
                f"discount_pct {p.discount_pct} is not allowed; for this customer (contract {profile['contract']}, "
                f"risk {float(profile['p_churn']):.2f}, monthly ${float(profile['monthly_charges']):.2f}) "
                f"it must be between 1 and {cap}")))
        if not 1 <= p.duration_months <= MAX_DISCOUNT_MONTHS:
            v.append(Violation("discount_cap", f"duration_months {p.duration_months} must be between 1 and {MAX_DISCOUNT_MONTHS}"))
    elif p.discount_pct != o["fixed_pct"] or p.duration_months != o["fixed_months"]:
        v.append(Violation("terms", (f"{p.offer_id} has fixed terms: discount_pct {o['fixed_pct']} and "
                                     f"duration_months {o['fixed_months']}")))
    if p.discount_pct * float(profile["monthly_charges"]) / 100 > ABS_DISCOUNT_CAP_PER_MONTH + 1e-9:
        v.append(Violation("discount_cap", f"discount would exceed the ${ABS_DISCOUNT_CAP_PER_MONTH:.0f} per month absolute cap"))
    return v


def _message_violations(p: Proposal) -> list[Violation]:
    v, msg, o = [], p.message, OFFERS[p.offer_id]
    limit = SMS_MAX_CHARS if p.channel == "sms" else EMAIL_MAX_CHARS
    if len(msg) > limit:
        v.append(Violation("length", f"{p.channel} message is {len(msg)} characters; the limit is {limit}. "
                                     f"Rewrite it in at most {limit - 70} characters"))
    if o["promotional"] and OPT_OUT_LINE.lower() not in msg.lower():
        v.append(Violation("opt_out_line", f'add this exact sentence at the end of the message: "{OPT_OUT_LINE}."'))
    for rule, patterns in (("forbidden_language", FORBIDDEN_MARKETING), ("prediction_terms", PREDICTION_TERMS),
                           ("protected_terms", PROTECTED_TERMS)):
        found = _hits(patterns, msg)
        if found:
            quotes = "; ".join(f'"{_snippet(msg, re.escape(w))}"' for w in found[:2])
            v.append(Violation(rule, f"remove these words or phrases from the message: {found} (found in: {quotes})"))
    if not o["promotional"]:
        found = _hits(PROMO_WORDS, msg)
        if found:
            v.append(Violation("promo_in_service_message", f"a service check-in must not contain promotional language: {found}"))
    if p.offer_id != "TECH_SUPPORT_TRIAL" and re.search(r"\bfree\b", msg, flags=re.I):
        v.append(Violation("forbidden_language", 'the word "free" may only be used for TECH_SUPPORT_TRIAL'))
    # Numbers must be exactly the offer terms (and nothing else).
    allowed = {"LOYALTY_DISCOUNT": {str(p.discount_pct), str(p.duration_months)},
               "TERM_UPGRADE": {"10", "12"}, "TECH_SUPPORT_TRIAL": {"3"}, "SERVICE_CHECKIN": set()}[p.offer_id]
    found_nums = set(re.findall(r"\d+(?:\.\d+)?", msg))
    if found_nums - allowed:
        extra = sorted(found_nums - allowed)
        quotes = "; ".join(f'"{_snippet(msg, rf"(?<![0-9.]){re.escape(n)}(?![0-9])")}"' for n in extra[:2])
        v.append(Violation("numbers", f"the message contains numbers that are not offer terms: {extra}. "
                                      f"Remove them (found in: {quotes})"))
    if allowed - found_nums:
        v.append(Violation("numbers", f"the message must state the offer terms; missing numbers: {sorted(allowed - found_nums)}"))
    return v


def check(p: Proposal, profile: dict, valid_ids: set[str] | None = None,
          check_citations: bool = True) -> list[Violation]:
    """All policy violations of a drafted proposal for this customer (empty list = compliant).

    check_citations=False is used when no policy text was shown to the model (nothing to cite)."""
    cites = (lambda: _citation_violations(p, profile, valid_ids)) if check_citations else (lambda: [])
    limit_reached = contact_limit_reached(profile)
    if p.action == "NO_ACTION":
        v = []
        if not limit_reached:
            v.append(Violation("no_action", "NO_ACTION is only allowed when the contact limit is reached; "
                                            "SERVICE_CHECKIN is always available, so choose an offer"))
        if p.offer_id != "NONE":
            v.append(Violation("terms", "NO_ACTION must use offer_id NONE"))
        return v + cites()
    if limit_reached:
        return [Violation("contact_limit", (f"customer already had {profile['contacts_last_90d']} contacts in 90 days "
                                            f"(limit {MAX_CONTACTS_90D}); the action must be NO_ACTION: offer_id NONE, discount_pct 0, "
                                            "duration_months 0 and an empty message"))] + cites()
    if p.offer_id not in OFFERS:
        return [Violation("terms", "action OFFER needs a real offer_id from the catalog")]
    v = []
    why = ineligibility(p.offer_id, profile)
    if why:
        hint = (" Choose a different offer this customer is eligible for (SERVICE_CHECKIN is always available)."
                if why[0] == "eligibility" else "")
        v.append(Violation(why[0], why[1] + hint))
    return v + _terms_violations(p, profile) + _message_violations(p) + cites()


def required_sections(profile: dict) -> set[str]:
    """Policy sections a correct decision for this customer depends on (used to score retrieval context recall)."""
    from .policy_spec import slug
    internet = "no internet phone only" if profile["internet_service"] == "No" else profile["internet_service"]
    need = {f"segment_playbooks#{slug(profile['contract'] + ' ' + internet)}",
            "message_standards#numbers-must-match-the-offer", "message_standards#prohibited-language"}
    oid = preferred_offer(profile)
    if oid == "NONE":
        return need | {"consent_and_contact#contact-frequency"}
    need.add(offer_section_id(oid))
    if OFFERS[oid]["promotional"]:
        need.add("consent_and_contact#required-opt-out-line")
    if oid == "LOYALTY_DISCOUNT":
        need |= {f"discount_caps#{slug(profile['contract'] + ' contracts')}", "discount_caps#risk-tiers"}
    if not int(profile["marketing_opt_in"]):
        need.add("consent_and_contact#marketing-consent")
    return need
