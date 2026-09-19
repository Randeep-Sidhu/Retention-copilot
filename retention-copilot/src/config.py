"""Central configuration: paths, seeds, column roles and the (clearly labelled) business assumptions."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = ROOT / "db" / "retention.db"          # analytics store (read-only for the agent)
RUNS_DB_PATH = ROOT / "db" / "runs.db"          # agent run log + human review queue (writable)
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

# IBM's public Telco churn sample (hosted in IBM's GitHub organisation).
DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")
RAW_CSV = DATA_DIR / "Telco-Customer-Churn.csv"

SEED = 42
TEST_SIZE = 0.25

RENAME = {
    "customerID": "customer_id", "gender": "gender", "SeniorCitizen": "senior_citizen",
    "Partner": "partner", "Dependents": "dependents", "tenure": "tenure_months",
    "PhoneService": "phone_service", "MultipleLines": "multiple_lines",
    "InternetService": "internet_service", "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup", "DeviceProtection": "device_protection",
    "TechSupport": "tech_support", "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies", "Contract": "contract",
    "PaperlessBilling": "paperless_billing", "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges", "TotalCharges": "total_charges", "Churn": "churn",
}

# Demographic attributes: excluded from the model AND from the agent-visible data.
# They are kept only in an audit table so we can check outcomes by group afterwards.
PROTECTED = ["gender", "senior_citizen", "partner", "dependents"]

# Tenure churns non-linearly (56% churn in months 0-3 vs 9% after 48), so the model uses bands (see model card).
TENURE_BINS = [-1, 3, 12, 24, 48, 1000]
TENURE_LABELS = ["0-3", "4-12", "13-24", "25-48", "49+"]

# Model inputs: account, service and billing behaviour only.
FEATURES = [
    "tenure_band", "phone_service", "multiple_lines", "internet_service", "online_security",
    "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "contract", "paperless_billing", "payment_method", "monthly_charges",
]
# Extra columns stored in the database for the agent's context (not model inputs).
DB_EXTRA = ["tenure_months"]
# Deliberately NOT a model input: ~perfectly collinear with tenure x monthly charges (see model card).
DROPPED_COLLINEAR = "total_charges"

# Simulated fields (NOT in the source dataset). They exist so that compliance rules
# (marketing consent, contact frequency) have something to test. Never used as model inputs.
SIMULATED_FIELDS = ["marketing_opt_in", "contacts_last_90d"]


@dataclass(frozen=True)
class Economics:
    """ASSUMPTIONS, not measurements.

    The dataset has no treatment data (nobody was randomly offered a discount), so nothing here
    is a causal estimate. We state assumptions and show how conclusions move when they change.
    """
    save_rate: float = 0.25            # P(stays | would have churned, contacted with an offer)
    takeup_non_churners: float = 0.25  # P(accepts discount | would have stayed anyway) -> cannibalisation
    discount_pct: float = 0.15         # standard offer depth
    discount_months: int = 6           # how long the discount lasts
    margin: float = 0.35               # contribution margin on monthly charges
    saved_lifetime_months: int = 12    # extra lifetime months gained if the customer is saved
    contact_cost: float = 3.0          # cost per contact (channel + reviewer time)
