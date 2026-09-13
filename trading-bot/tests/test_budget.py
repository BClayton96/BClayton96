from datetime import date, timedelta

from src.budget import WeeklyBudget, _week_start


def test_fresh_budget_has_full_remaining(tmp_path):
    budget = WeeklyBudget(tmp_path / "budget.json", weekly_limit=50.0)
    assert budget.remaining() == 50.0


def test_record_spend_reduces_remaining(tmp_path):
    budget = WeeklyBudget(tmp_path / "budget.json", weekly_limit=50.0)
    budget.record_spend(20.0)
    assert budget.remaining() == 30.0


def test_cannot_go_negative(tmp_path):
    budget = WeeklyBudget(tmp_path / "budget.json", weekly_limit=50.0)
    budget.record_spend(60.0)
    assert budget.remaining() == 0.0


def test_state_persists_across_instances(tmp_path):
    path = tmp_path / "budget.json"
    WeeklyBudget(path, weekly_limit=50.0).record_spend(15.0)
    reloaded = WeeklyBudget(path, weekly_limit=50.0)
    assert reloaded.remaining() == 35.0


def test_new_week_resets_spend(tmp_path):
    path = tmp_path / "budget.json"
    budget = WeeklyBudget(path, weekly_limit=50.0)
    budget.record_spend(40.0)
    assert budget.remaining() == 10.0

    # simulate a week having passed by rewriting the stored week_start
    last_week = (_week_start(date.today()) - timedelta(days=7)).isoformat()
    import json

    data = json.loads(path.read_text())
    data["week_start"] = last_week
    path.write_text(json.dumps(data))

    rolled = WeeklyBudget(path, weekly_limit=50.0)
    assert rolled.remaining() == 50.0
