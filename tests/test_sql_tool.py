import pytest

from src.sql_tool import SqlNotAllowed, SqlTool


def test_profile_lookup_through_the_view(test_db):
    p = SqlTool(test_db).profile("A-FIBER")
    assert p["contract"] == "Month-to-month" and p["p_churn"] == 0.85
    assert SqlTool(test_db).profile("nope") is None


def test_aggregate_queries_through_the_view_are_allowed(test_db):
    rows = SqlTool(test_db).query("SELECT contract, COUNT(*) AS n FROM v_customer_profile GROUP BY contract ORDER BY contract")
    assert sum(r["n"] for r in rows) == 6


def test_view_exposes_no_label_and_no_demographics(test_db):
    row = SqlTool(test_db).query("SELECT * FROM v_customer_profile LIMIT 1")[0]
    assert not {"churn_actual", "split", "gender", "senior_citizen", "partner", "dependents"} & set(row)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM labels", "SELECT * FROM audit_attributes", "SELECT * FROM customers", "SELECT * FROM scores",
    "SELECT customer_id FROM v_customer_profile UNION SELECT customer_id FROM labels",
    "SELECT (SELECT churn_actual FROM labels LIMIT 1) FROM v_customer_profile",
    "SELECT name FROM sqlite_master", "DROP TABLE customers", "SELECT 1; DROP TABLE customers",
    "PRAGMA table_info(labels)", "ATTACH DATABASE 'x.db' AS x", "INSERT INTO customers VALUES (1)",
    "UPDATE customers SET monthly_charges = 0", "SELECT 1 -- comment",
])
def test_everything_else_is_blocked_by_the_database_not_the_prompt(test_db, sql):
    with pytest.raises(SqlNotAllowed):
        SqlTool(test_db).query(sql)


def test_connection_is_read_only(test_db):
    tool = SqlTool(test_db)
    with pytest.raises(SqlNotAllowed):
        tool.query("WITH x AS (SELECT 1) DELETE FROM customers")
    assert len(tool.query("SELECT * FROM v_customer_profile")) == 6


def test_row_limit(test_db):
    assert len(SqlTool(test_db, max_rows=2).query("SELECT * FROM v_customer_profile")) == 2
