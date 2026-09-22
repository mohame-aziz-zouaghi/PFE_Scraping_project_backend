from app.core.cost_tracker import estimate_cost_cents


def test_cost_gpt4o_mini():
    cost = estimate_cost_cents("gpt-4o-mini", 1_000_000, 0)
    assert cost == 15.0


def test_cost_embedding():
    cost = estimate_cost_cents("text-embedding-3-small", 1_000_000, 0)
    assert cost == 2.0


def test_cost_unknown_model_returns_zero():
    assert estimate_cost_cents("nonexistent", 1000, 1000) == 0.0


def test_cost_ollama_is_free():
    assert estimate_cost_cents("llama3.1:8b", 10000, 5000) == 0.0
