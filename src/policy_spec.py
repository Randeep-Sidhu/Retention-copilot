"""Single source of truth for the (fictional) retention policy.

The markdown documents in policies/ are GENERATED from this file (python -m src.policy_docs), and the
deterministic verifier in policy_engine.py reads the same constants. The text the LLM retrieves and the rules
that check its output therefore cannot drift apart. This is a made-up company policy for a portfolio project,
not real telecom guidance.
"""
import re

COMPANY = "Maple Telecom"

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["Fiber optic", "DSL", "No"]          # values of internet_service in the data ("No" = phone only)

MAX_CONTACTS_90D = 3            # outreach is blocked once a customer has had this many contacts in 90 days
SMS_MAX_CHARS = 320
EMAIL_MAX_CHARS = 900
OPT_OUT_LINE = "Reply STOP to opt out"

NEW_CUSTOMER_MAX_TENURE = 3     # months 0-3 = "new customer"
HIGH_VALUE_MONTHLY = 100.0      # monthly charge above which discount offers get priority human review
ABS_DISCOUNT_CAP_PER_MONTH = 15.0
HIGH_RISK_THRESHOLD = 0.60      # churn score at or above this = high-risk tier
STANDARD_DISCOUNT_PCT = 10
MAX_DISCOUNT_PCT = {"Month-to-month": 15, "One year": 10, "Two year": 0}
MAX_DISCOUNT_MONTHS = 6

OFFERS = {
    "LOYALTY_DISCOUNT": dict(title="Loyalty discount", promotional=True,
                             contracts=["Month-to-month", "One year"], min_tenure=NEW_CUSTOMER_MAX_TENURE + 1,
                             needs_internet=False, needs_no_tech_support=False),
    "TERM_UPGRADE": dict(title="Term upgrade", promotional=True, contracts=["Month-to-month"],
                         min_tenure=NEW_CUSTOMER_MAX_TENURE + 1, needs_internet=False, needs_no_tech_support=False,
                         fixed_pct=10, fixed_months=12),
    "TECH_SUPPORT_TRIAL": dict(title="Tech support trial", promotional=True, contracts=list(CONTRACTS),
                               min_tenure=0, needs_internet=True, needs_no_tech_support=True,
                               fixed_pct=0, fixed_months=3),
    "SERVICE_CHECKIN": dict(title="Service check-in", promotional=False, contracts=list(CONTRACTS),
                            min_tenure=0, needs_internet=False, needs_no_tech_support=False,
                            fixed_pct=0, fixed_months=0),
}

# Preferred order of offers per (contract, internet service). The verifier does NOT enforce the order;
# the evaluation reports how often the drafted offer matches the first eligible entry.
LADDER = {
    ("Month-to-month", "Fiber optic"): ["TECH_SUPPORT_TRIAL", "TERM_UPGRADE", "LOYALTY_DISCOUNT", "SERVICE_CHECKIN"],
    ("Month-to-month", "DSL"): ["TERM_UPGRADE", "TECH_SUPPORT_TRIAL", "LOYALTY_DISCOUNT", "SERVICE_CHECKIN"],
    ("Month-to-month", "No"): ["TERM_UPGRADE", "LOYALTY_DISCOUNT", "SERVICE_CHECKIN"],
    ("One year", "Fiber optic"): ["TECH_SUPPORT_TRIAL", "LOYALTY_DISCOUNT", "SERVICE_CHECKIN"],
    ("One year", "DSL"): ["TECH_SUPPORT_TRIAL", "LOYALTY_DISCOUNT", "SERVICE_CHECKIN"],
    ("One year", "No"): ["LOYALTY_DISCOUNT", "SERVICE_CHECKIN"],
    ("Two year", "Fiber optic"): ["TECH_SUPPORT_TRIAL", "SERVICE_CHECKIN"],
    ("Two year", "DSL"): ["TECH_SUPPORT_TRIAL", "SERVICE_CHECKIN"],
    ("Two year", "No"): ["SERVICE_CHECKIN"],
}

# ---- message lint vocabularies (regular expressions, matched case-insensitively) ----
FORBIDDEN_MARKETING = [r"\bguarantee[sd]?\b", r"\bact now\b", r"\blast chance\b", r"\blimited[- ]time\b",
                       r"\bhurry\b", r"\bdon'?t miss\b", r"\burgent(ly)?\b"]
PREDICTION_TERMS = [r"\bchurn\b", r"\bcancel(l?ing|l?ation|s|led)?\b", r"\bleav(e|ing)\b", r"\bat risk\b",
                    r"\blikely to\b", r"\bcompetitors?\b", r"\bswitch(ing)? (to|providers?|carriers?)\b",
                    r"\bscore\b", r"\bpredict(ed|ion|s)?\b"]
PROTECTED_TERMS = [r"\bage\b", r"\baged\b", r"\bsenior(s)?\b", r"\belderly\b", r"\bgender\b", r"\bwoman\b",
                   r"\bmarried\b", r"\bspouse\b", r"\bwife\b", r"\bhusband\b", r"\bfamily\b", r"\bchildren\b",
                   r"\bkids?\b"]
PROMO_WORDS = [r"\bdiscount\b", r"\boffer\b", r"\bsave\b", r"\bsavings?\b", r"\bdeal\b", r"\bfree\b",
               r"%", r"\bpromo(tion(al)?)?\b"]
PROHIBITED_PHRASES_FOR_DOCS = ["guarantee / guaranteed", "act now", "last chance", "limited time", "hurry",
                               "don't miss", "urgent"]


def slug(text: str) -> str:
    """Section heading -> id fragment, e.g. 'Service check-in' -> 'service-check-in'."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def offer_section_id(offer_id: str) -> str:
    return f"offer_catalog#{slug(OFFERS[offer_id]['title'])}"
