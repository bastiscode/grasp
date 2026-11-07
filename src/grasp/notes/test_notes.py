import os

import pytest

import grasp.notes as notes_module
from grasp.configs import NotesFromExplorationConfig
from grasp.utils import is_model_error


def error_output(reason: str) -> dict:
    return {
        "type": "output",
        "output": None,
        "error": {"content": f"{reason} error", "reason": reason},
        "messages": [],
    }


def test_is_model_error():
    assert is_model_error(error_output("api"))
    assert is_model_error(error_output("timeout"))
    assert is_model_error(error_output("empty"))
    # limits and loops come from the agent, not from the model API
    assert not is_model_error(error_output("limit"))
    assert not is_model_error(error_output("loop"))
    assert not is_model_error({"output": {}, "error": None})


def run_exploration(monkeypatch, tmp_path, outputs: list[dict]) -> str:
    monkeypatch.setattr(notes_module, "setup", lambda config: ([], {}))
    monkeypatch.setattr(notes_module, "load_example_indices", lambda *args: {})
    monkeypatch.setattr(notes_module, "load_notes", lambda config: ([], {}))
    monkeypatch.setattr(notes_module, "generate", lambda *args, **kwargs: None)
    rounds = iter(outputs)
    monkeypatch.setattr(notes_module, "consume_generator", lambda gen: next(rounds))

    out_dir = str(tmp_path / "exploration")
    config = NotesFromExplorationConfig(mode="functional", num_rounds=len(outputs))
    notes_module.take_notes_from_exploration(config, out_dir)
    return out_dir


def test_exploration_stops_on_model_error(monkeypatch, tmp_path):
    ok = {"type": "output", "output": {}, "error": None, "messages": []}
    with pytest.raises(RuntimeError, match="round 1 failed with a model error"):
        run_exploration(monkeypatch, tmp_path, [ok, error_output("api"), ok])

    out_dir = str(tmp_path / "exploration")
    # the failed round keeps its trace but writes no notes, the third never runs
    assert os.path.exists(os.path.join(out_dir, "output.exploration.round_1.json"))
    assert not os.path.exists(os.path.join(out_dir, "notes.exploration.round_1.json"))
    assert not os.path.exists(os.path.join(out_dir, "output.exploration.round_2.json"))


def test_exploration_continues_after_limit(monkeypatch, tmp_path):
    ok = {"type": "output", "output": {}, "error": None, "messages": []}
    out_dir = run_exploration(monkeypatch, tmp_path, [error_output("limit"), ok])
    assert os.path.exists(os.path.join(out_dir, "notes.exploration.round_1.json"))
