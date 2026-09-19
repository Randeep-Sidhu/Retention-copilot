"""Policy engine, verifier, reference drafter and generated documents. Offline; no LLM."""
import random

import pytest

from src import policy_docs as PD
from src import policy_spec as S
from src.policy_engine import check, eligible_offers, max_loyalty_pct, preferred_offer, review_flags
from src.reference_drafter import reference_proposal
from src.retriever import load_corpus
from src.schemas import PROPOSAL_SCHEMA, Proposal

VALID_IDS = {c.id for c in load_corpus()}


def profile(**kw):
    base = dict(customer_id="X", contract="Month-to-month", internet_service="Fiber optic", tech_support="No",
                tenure_months=10, tenure_band="4-12", monthly_charges=80.0, marketing_opt_in=1,
                contacts_last_90d=0, p_churn=0.5)
    base.update(kw)
    return base


def proposal(**kw):
    base = dict(action="OFFER", offer_id="LOYALTY_DISCOUNT", discount_pct=10, duration_months=6, channel="sms",
                message=("Hello, thank you for being with Maple Telecom. We would like to offer you 10% off your monthly "
                         "charges for 6 months. Reply YES to accept. Reply STOP to opt out."),
                rationale="test", citations=["offer_catalog#loyalty-discount", "discount_caps#month-to-month-contracts"])
    base.update(kw)
    return Proposal(**base)


def rules(p, prof, **kw):
    return {v.rule for v in check(p, prof, VALID_IDS)}


# ------------------------------------------------------------------ the policy is satisfiable
def random_profile(rng):
    internet = rng.choice(S.INTERNET)
    return profile(contract=rng.choice(S.CONTRACTS), internet_service=internet,
                   tech_support="No internet service" if internet == "No" else rng.choice(["Yes", "No"]),
                   tenure_months=rng.randint(0, 72), monthly_charges=round(rng.uniform(18, 120), 2),
                   marketing_opt_in=rng.choice([0, 1, 1, 1]), contacts_last_90d=rng.randint(0, 4),
                   p_churn=round(rng.random(), 3))


def test_reference_drafter_is_compliant_for_every_kind_of_customer():
    rng = random.Random(0)
    for _ in range(1500):
        prof = random_profile(rng)
        assert check(reference_proposal(prof), prof, VALID_IDS) == [], prof


def test_reference_drafter_covers_all_offers_and_no_action():
    rng = random.Random(1)
    chosen = {reference_proposal(random_profile(rng)).offer_id for _ in range(1500)}
    assert chosen == {"LOYALTY_DISCOUNT", "TERM_UPGRADE", "TECH_SUPPORT_TRIAL", "SERVICE_CHECKIN", "NONE"}


# ------------------------------------------------------------------ eligibility, tiers, ladder
def test_ladders_cover_every_segment_and_end_with_the_always_available_offer():
    assert set(S.LADDER) == {(c, i) for c in S.CONTRACTS for i in S.INTERNET}
    assert all(ladder[-1] == "SERVICE_CHECKIN" for ladder in S.LADDER.values())


@pytest.mark.parametrize("kw,expected", [
    (dict(), "TECH_SUPPORT_TRIAL"),
    (dict(tech_support="Yes"), "TERM_UPGRADE"),
    (dict(contract="One year", tech_support="Yes"), "LOYALTY_DISCOUNT"),
    (dict(marketing_opt_in=0), "SERVICE_CHECKIN"),
    (dict(contacts_last_90d=3), "NONE"),
    (dict(tenure_months=2, tech_support="Yes"), "SERVICE_CHECKIN"),
    (dict(contract="Two year", internet_service="No", tech_support="No internet service"), "SERVICE_CHECKIN"),
    (dict(internet_service="DSL"), "TERM_UPGRADE"),
])
def test_preferred_offer(kw, expected):
    assert preferred_offer(profile(**kw)) == expected


def test_eligible_offers_edge_cases():
    assert eligible_offers(profile(contacts_last_90d=3)) == []
    assert eligible_offers(profile(marketing_opt_in=0)) == ["SERVICE_CHECKIN"]
    assert "LOYALTY_DISCOUNT" not in eligible_offers(profile(contract="Two year"))
    assert "TECH_SUPPORT_TRIAL" not in eligible_offers(profile(internet_service="No", tech_support="No internet service"))


@pytest.mark.parametrize("kw,expected", [
    (dict(p_churn=0.3), 10), (dict(p_churn=0.7), 15), (dict(p_churn=0.7, contract="One year"), 10),
    (dict(p_churn=0.7, monthly_charges=120.0), 12), (dict(p_churn=0.7, monthly_charges=100.0), 15),
    (dict(contract="Two year"), 0),
])
def test_max_loyalty_pct(kw, expected):
    assert max_loyalty_pct(profile(**kw)) == expected


# ------------------------------------------------------------------ the verifier catches each violation type
def test_compliant_proposal_passes():
    assert check(proposal(), profile(), VALID_IDS) == []


HIGH_MSG = ("Hello, thank you for being with Maple Telecom. We would like to offer you 15% off your monthly charges "
            "for 6 months. Reply YES to accept. Reply STOP to opt out.")


@pytest.mark.parametrize("case,prof,prop,rule", [
    ("cap by contract", profile(contract="One year", p_churn=0.9), proposal(discount_pct=15, message=HIGH_MSG), "discount_cap"),
    ("standard tier cap", profile(p_churn=0.3), proposal(discount_pct=15, message=HIGH_MSG), "discount_cap"),
    ("absolute $ cap", profile(p_churn=0.9, monthly_charges=120.0), proposal(discount_pct=15, message=HIGH_MSG), "discount_cap"),
    ("too many months", profile(), proposal(duration_months=9, message=proposal().message.replace("6", "9")), "discount_cap"),
    ("no consent", profile(marketing_opt_in=0), proposal(), "consent"),
    ("contact limit", profile(contacts_last_90d=3), proposal(), "contact_limit"),
    ("lazy no-action", profile(), proposal(action="NO_ACTION", offer_id="NONE", discount_pct=0, duration_months=0, message=""), "no_action"),
    ("two-year discount", profile(contract="Two year"), proposal(), "eligibility"),
    ("new customer discount", profile(tenure_months=2), proposal(), "eligibility"),
    ("has tech support", profile(tech_support="Yes"), proposal(offer_id="TECH_SUPPORT_TRIAL", discount_pct=0, duration_months=3,
        message="Hello, we will add Tech Support at no charge for 3 months. Reply YES to accept. Reply STOP to opt out.",
        citations=["offer_catalog#tech-support-trial"]), "eligibility"),
    ("wrong fixed terms", profile(), proposal(offer_id="TERM_UPGRADE", discount_pct=15, duration_months=12,
        message="Hello, move to a 12-month term for 15% off for 12 months. Reply STOP to opt out.",
        citations=["offer_catalog#term-upgrade"]), "terms"),
    ("no opt-out line", profile(), proposal(message=proposal().message.replace(" Reply STOP to opt out.", "")), "opt_out_line"),
    ("guarantee", profile(), proposal(message=proposal().message.replace("Hello,", "Hello, guaranteed savings!")), "forbidden_language"),
    ("cancel", profile(), proposal(message=proposal().message.replace("Hello,", "Hello, before you cancel,")), "prediction_terms"),
    ("senior", profile(), proposal(message=proposal().message.replace("Hello,", "Hello senior customer,")), "protected_terms"),
    ("wrong number", profile(), proposal(message=proposal().message.replace("10%", "20%")), "numbers"),
    ("terms not stated", profile(), proposal(message="Hello, we value you. Reply STOP to opt out."), "numbers"),
    ("sms too long", profile(), proposal(message=proposal().message + " " + "x" * 300), "length"),
    ("free outside trial", profile(), proposal(message=proposal().message.replace("Hello,", "Hello, it is free.")), "forbidden_language"),
    ("promo words in service message", profile(), proposal(offer_id="SERVICE_CHECKIN", discount_pct=0, duration_months=0,
        message="Hello, a discount is waiting for you.", citations=["offer_catalog#service-check-in"]), "promo_in_service_message"),
    ("number in service message", profile(), proposal(offer_id="SERVICE_CHECKIN", discount_pct=0, duration_months=0,
        message="Hello, call us within 2 days.", citations=["offer_catalog#service-check-in"]), "numbers"),
    ("unknown citation", profile(), proposal(citations=["offer_catalog#loyalty-discount", "made_up#section"]), "citations"),
    ("missing catalog citation", profile(), proposal(citations=["discount_caps#month-to-month-contracts"]), "citations"),
    ("no citations", profile(), proposal(citations=[]), "citations"),
])
def test_verifier_flags(case, prof, prop, rule):
    assert rule in rules(prop, prof), case


def test_no_action_is_correct_when_contact_limit_reached():
    prop = proposal(action="NO_ACTION", offer_id="NONE", discount_pct=0, duration_months=0, message="",
                    citations=["consent_and_contact#contact-frequency"])
    assert check(prop, profile(contacts_last_90d=3), VALID_IDS) == []
    assert "citations" in rules(prop.model_copy(update={"citations": ["escalation#human-approval"]}), profile(contacts_last_90d=4))


def test_review_flags():
    assert review_flags(profile(monthly_charges=110.0), proposal()) == ["high_value_account"]
    assert review_flags(profile(tenure_months=1), proposal()) == ["new_customer"]
    assert review_flags(profile(), proposal()) == []


# ------------------------------------------------------------------ schema and documents
def test_schema_is_flat_enough_for_constrained_decoding():
    text = str(PROPOSAL_SCHEMA)
    assert "anyOf" not in text and "null" not in text
    assert set(PROPOSAL_SCHEMA["required"]) == set(PROPOSAL_SCHEMA["properties"])


def test_policy_documents_on_disk_match_the_spec():
    for name, render in PD.DOCS.items():
        assert (PD.POLICY_DIR / name).read_text(encoding="utf-8") == render().rstrip() + "\n", (
            f"{name} is stale: run python -m src.policy_docs")


def test_documents_state_the_numbers_the_verifier_enforces():
    text = " ".join(c.text for c in load_corpus())
    for needle in (f"{S.MAX_DISCOUNT_PCT['Month-to-month']}%", f"{S.MAX_DISCOUNT_PCT['One year']}%",
                   f"${S.ABS_DISCOUNT_CAP_PER_MONTH:.0f}", str(S.MAX_CONTACTS_90D), f"{S.SMS_MAX_CHARS} characters",
                   S.OPT_OUT_LINE, f"{S.HIGH_RISK_THRESHOLD:.2f}"):
        assert needle in text, needle


# ------------------------------------------------------------------ feedback quality (what a small model actually sees)
REAL_DRAFT = ("Dear Customer,\n\nWe appreciate your continued choice of Maple Telecom. As a valued customer with 17 months of "
              "tenure, we are offering you a 10% discount on your monthly charges for up to 6 months. Thank you for being "
              "part of the Maple Telecom family. Reply STOP to opt out at any time.")


def test_violation_details_quote_the_offending_text_and_give_a_length_target():
    prop = proposal(message=REAL_DRAFT + " " + "Thank you for choosing us. " * 8)          # also over the SMS limit
    found = {v.rule: v.detail for v in check(prop, profile(), VALID_IDS)}
    assert "17 months" in found["numbers"] and "found in" in found["numbers"]
    assert "family" in found["protected_terms"] and "Maple Telecom family" in found["protected_terms"]
    assert "at most 250 characters" in found["length"]


def test_revision_request_asks_for_a_full_rewrite_only_for_message_problems():
    from src.prompts import revision_request
    assert "from scratch" in revision_request(["[numbers] contains 17", "[length] too long"])
    assert "from scratch" not in revision_request(["[discount_cap] discount_pct 15 is not allowed"])
    assert "Change only what is needed" in revision_request(["[eligibility] not available"])


def test_schema_puts_the_reasoning_field_first():
    assert list(PROPOSAL_SCHEMA["properties"])[0] == "rationale"
