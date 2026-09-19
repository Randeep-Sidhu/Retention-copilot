"""A deterministic, policy-perfect drafter. Used as (1) a test oracle proving the policy is satisfiable,
(2) the offline 'stub' LLM for CI and demos, and (3) an upper-bound reference in the evaluation."""
from __future__ import annotations

from .policy_engine import max_loyalty_pct, preferred_offer
from .policy_spec import COMPANY, OPT_OUT_LINE, STANDARD_DISCOUNT_PCT, offer_section_id
from .schemas import Proposal

_OPT = f" {OPT_OUT_LINE}."


def reference_proposal(profile: dict) -> Proposal:
    oid = preferred_offer(profile)
    if oid == "NONE":
        return Proposal(action="NO_ACTION", offer_id="NONE", discount_pct=0, duration_months=0, channel="sms",
                        message="", rationale="The customer has reached the 90-day contact limit, so no outreach is sent.",
                        citations=["consent_and_contact#contact-frequency"])
    pct, months = 0, 0
    if oid == "LOYALTY_DISCOUNT":
        pct, months = min(STANDARD_DISCOUNT_PCT, max_loyalty_pct(profile)), 6
        msg = (f"Hello, thank you for being with {COMPANY}. We would like to offer you {pct}% off your monthly "
               f"charges for {months} months. Reply YES to accept.{_OPT}")
    elif oid == "TERM_UPGRADE":
        pct, months = 10, 12
        msg = (f"Hello, thank you for staying with {COMPANY}. Move to a 12-month term and enjoy 10% off your monthly "
               f"charges for 12 months. Reply YES to accept.{_OPT}")
    elif oid == "TECH_SUPPORT_TRIAL":
        pct, months = 0, 3
        msg = f"Hello, we would like to add Tech Support to your plan at no charge for 3 months. Reply YES to accept.{_OPT}"
    else:
        msg = (f"Hello, we would love to make sure your plan still fits your needs. A {COMPANY} adviser can call you "
               "at a time that suits you. Reply CALL and we will arrange it.")
    cites = [offer_section_id(oid), "segment_playbooks#" + _segment_slug(profile)]
    return Proposal(action="OFFER", offer_id=oid, discount_pct=pct, duration_months=months, channel="sms",
                    message=msg, rationale=f"{oid} is the first eligible offer in the playbook for this segment.",
                    citations=cites)


def _segment_slug(profile: dict) -> str:
    from .policy_spec import slug
    internet = "no internet phone only" if profile["internet_service"] == "No" else profile["internet_service"]
    return slug(f"{profile['contract']} {internet}")
