"""Shared fixtures: a tiny analytics database with the SAME schema that src.phase1 writes."""
import json
import sqlite3

import pytest

DRIVERS = json.dumps([{"feature": "tenure_band", "value": "4-12", "log_odds": 0.9},
                      {"feature": "contract", "value": "Month-to-month", "log_odds": 0.5}])
PROTECT = json.dumps([{"feature": "phone_service", "value": "Yes", "log_odds": -0.05}])

# customer_id, contract, internet, tech_support, tenure, monthly, opt_in, contacts, p_churn
CUSTOMERS = [
    ("A-FIBER", "Month-to-month", "Fiber optic", "No", 5, 89.15, 1, 0, 0.85),      # -> TECH_SUPPORT_TRIAL
    ("B-ONEYR", "One year", "DSL", "Yes", 20, 60.0, 1, 0, 0.45),                   # -> LOYALTY_DISCOUNT
    ("C-OPTOUT", "Month-to-month", "DSL", "No", 12, 55.0, 0, 0, 0.70),             # -> SERVICE_CHECKIN
    ("D-LIMIT", "Month-to-month", "Fiber optic", "No", 12, 90.0, 1, 3, 0.80),      # -> NO_ACTION
    ("E-LOWRISK", "Two year", "DSL", "Yes", 60, 45.0, 1, 0, 0.03),                 # gate skips
    ("F-2YR", "Two year", "No", "No internet service", 40, 25.0, 1, 0, 0.60),      # -> SERVICE_CHECKIN
]


@pytest.fixture()
def test_db(tmp_path):
    path = tmp_path / "retention.db"
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE customers (customer_id TEXT, tenure_band TEXT, phone_service TEXT, multiple_lines TEXT,
      internet_service TEXT, online_security TEXT, online_backup TEXT, device_protection TEXT, tech_support TEXT,
      streaming_tv TEXT, streaming_movies TEXT, contract TEXT, paperless_billing TEXT, payment_method TEXT,
      monthly_charges REAL, tenure_months INTEGER, marketing_opt_in INTEGER, contacts_last_90d INTEGER);
    CREATE TABLE scores (customer_id TEXT, p_churn REAL, drivers_json TEXT, protective_json TEXT, model TEXT);
    CREATE TABLE labels (customer_id TEXT, churn_actual INTEGER, split TEXT);
    CREATE TABLE audit_attributes (customer_id TEXT, gender TEXT, senior_citizen INTEGER, partner TEXT, dependents TEXT);
    CREATE VIEW v_customer_profile AS SELECT c.*, s.p_churn, s.drivers_json, s.protective_json
      FROM customers c JOIN scores s USING (customer_id);
    """)
    for cid, contract, net, tech, ten, mrr, opt, contacts, p in CUSTOMERS:
        band = "0-3" if ten <= 3 else "4-12" if ten <= 12 else "13-24" if ten <= 24 else "25-48" if ten <= 48 else "49+"
        con.execute("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, band, "Yes", "No", net, "No", "No", "No", tech, "No", "No", contract, "Yes",
                     "Electronic check", mrr, ten, opt, contacts))
        con.execute("INSERT INTO scores VALUES (?,?,?,?,?)", (cid, p, DRIVERS, PROTECT, "logreg"))
        con.execute("INSERT INTO labels VALUES (?,?,?)", (cid, 1, "test"))
        con.execute("INSERT INTO audit_attributes VALUES (?,?,?,?,?)", (cid, "Female", 0, "No", "No"))
    con.commit()
    con.close()
    return path
