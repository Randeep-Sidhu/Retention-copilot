"""Development / final evaluation pools: disjoint, deterministic, and free of customers inspected during development."""
import pytest

from src.config import DB_PATH
from src.scenarios import DEV_SEEN, _bucket, build_scenarios, gate_passing_test_ids, pool_ids, sample_test_customers


def test_bucket_is_stable_across_calls():
    assert {_bucket("7590-VHVEG") for _ in range(5)} == {_bucket("7590-VHVEG")}


def test_pools_are_disjoint_and_together_cover_the_gate_passing_customers(test_db):
    dev, final = set(pool_ids("dev", test_db)), set(pool_ids("final", test_db))
    assert dev.isdisjoint(final) and dev | final == set(gate_passing_test_ids(test_db))
    with pytest.raises(ValueError):
        pool_ids("other", test_db)


def test_smoke_test_customers_come_from_the_dev_pool_only(test_db):
    assert set(sample_test_customers(10, db_path=test_db)) <= set(pool_ids("dev", test_db))


def test_scenarios_are_deterministic_and_stay_inside_their_pool(test_db):
    for pool in ("dev", "final"):
        a, b = build_scenarios(5, db_path=test_db, pool=pool), build_scenarios(5, db_path=test_db, pool=pool)
        assert a == b and set(a) <= set(pool_ids(pool, test_db))


@pytest.mark.skipif(not DB_PATH.exists(), reason="run python -m src.phase1 first")
def test_real_final_pool_excludes_every_customer_inspected_during_development():
    final, dev = set(pool_ids("final")), set(pool_ids("dev"))
    assert final.isdisjoint(DEV_SEEN) and final.isdisjoint(dev)
    assert len(final) >= 100 and len(dev) >= 100
    assert len(set(build_scenarios(48, pool="final"))) == 48
