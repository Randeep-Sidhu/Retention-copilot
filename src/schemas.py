"""Structured output the agent must produce. Deliberately no Optional/anyOf: small local models and
constrained decoding handle flat enums much more reliably ("NONE" instead of null).

Field order matters: constrained decoding emits properties in this order, so `rationale` comes first and acts as a
short reasoning step before the model commits to an offer.
"""
from typing import Literal

from pydantic import BaseModel, Field

OfferId = Literal["LOYALTY_DISCOUNT", "TERM_UPGRADE", "TECH_SUPPORT_TRIAL", "SERVICE_CHECKIN", "NONE"]


class Proposal(BaseModel):
    rationale: str = Field(description="One or two sentences, written first: the customer's segment, the first "
                                       "eligible offer and why. Also shown to the human reviewer")
    action: Literal["OFFER", "NO_ACTION"]
    offer_id: OfferId
    discount_pct: int = Field(ge=0, le=100, description="Percent discount; 0 if the offer has none")
    duration_months: int = Field(ge=0, le=36, description="Months the discount or trial lasts; 0 if none")
    channel: Literal["sms", "email"]
    message: str = Field(description="Short SMS text; empty string when action is NO_ACTION")
    citations: list[str] = Field(description="Policy section ids used, e.g. offer_catalog#loyalty-discount")


PROPOSAL_SCHEMA = Proposal.model_json_schema()
