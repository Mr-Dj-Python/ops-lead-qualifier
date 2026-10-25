from pathlib import Path

import pytest

from main import main
from pipeline.config import ConfigError, load_config, load_env

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
DATA = ROOT / "data" / "leads_training.csv"
DEFAULTS = {"industry": 0.40, "size": 0.25, "source": 0.20, "recency": 0.15}
WEIGHT_VARS = ["WEIGHT_INDUSTRY", "WEIGHT_SIZE", "WEIGHT_SOURCE", "WEIGHT_RECENCY"]


@pytest.fixture
def clean_env(monkeypatch):
    for name in WEIGHT_VARS:
        monkeypatch.delenv(name, raising=False)


# --- load_config ---------------------------------------------------------------------------

def test_yaml_weights_are_used_when_no_env_is_set():
    config = load_config(CONFIG, env={})
    assert config["weights"] == DEFAULTS
    assert config["weights_overridden"] == []


def test_all_four_weights_can_be_overridden():
    env = {"WEIGHT_INDUSTRY": "0.5", "WEIGHT_SIZE": "0.2", "WEIGHT_SOURCE": "0.2", "WEIGHT_RECENCY": "0.1"}
    config = load_config(CONFIG, env=env)
    assert config["weights"] == {"industry": 0.5, "size": 0.2, "source": 0.2, "recency": 0.1}
    assert sorted(config["weights_overridden"]) == ["industry", "recency", "size", "source"]


def test_overriding_a_subset_keeps_the_yaml_values_for_the_rest():
    env = {"WEIGHT_INDUSTRY": "0.35", "WEIGHT_RECENCY": "0.20"}
    config = load_config(CONFIG, env=env)
    assert config["weights"] == {"industry": 0.35, "size": 0.25, "source": 0.20, "recency": 0.20}
    assert sorted(config["weights_overridden"]) == ["industry", "recency"]


def test_override_that_breaks_the_sum_is_rejected_and_shows_the_weights():
    with pytest.raises(ConfigError, match=r"add up to 1, got 1\.1.*industry 0\.5"):
        load_config(CONFIG, env={"WEIGHT_INDUSTRY": "0.5"})


def test_env_names_are_case_insensitive_after_the_prefix():
    config = load_config(CONFIG, env={"WEIGHT_industry": "0.40"})
    assert config["weights_overridden"] == ["industry"]


@pytest.mark.parametrize("value", ["lots", "", "0.4.0", "nan", "inf", "-0.1"])
def test_invalid_weight_values_are_rejected(value):
    with pytest.raises(ConfigError, match="WEIGHT_SIZE must be a number"):
        load_config(CONFIG, env={"WEIGHT_SIZE": value})


def test_unknown_weight_variable_is_rejected_to_catch_typos():
    with pytest.raises(ConfigError, match="unknown weight variable WEIGHT_RECENY.*WEIGHT_RECENCY"):
        load_config(CONFIG, env={"WEIGHT_RECENY": "0.15"})


def test_unrelated_env_variables_are_ignored():
    assert load_config(CONFIG, env={"PATH": "/bin", "WEIGHTS": "x", "LLM_API_KEY": "k"})["weights"] == DEFAULTS


def test_zero_weight_is_allowed():
    env = {"WEIGHT_RECENCY": "0", "WEIGHT_INDUSTRY": "0.55"}
    assert load_config(CONFIG, env=env)["weights"]["recency"] == 0


# --- load_env ------------------------------------------------------------------------------

def test_load_env_reads_the_file(clean_env, tmp_path):
    path = tmp_path / ".env"
    path.write_text("WEIGHT_SIZE=0.30\n# comment\nOTHER=1\n")
    env = load_env(path)
    assert env["WEIGHT_SIZE"] == "0.30" and env["OTHER"] == "1"


def test_real_environment_variable_beats_the_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("WEIGHT_SIZE=0.30\n")
    monkeypatch.setenv("WEIGHT_SIZE", "0.25")
    assert load_env(path)["WEIGHT_SIZE"] == "0.25"


def test_load_env_does_not_modify_the_process_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("ONLY_IN_FILE", raising=False)
    path = tmp_path / ".env"
    path.write_text("ONLY_IN_FILE=1\n")
    assert load_env(path)["ONLY_IN_FILE"] == "1"
    import os
    assert "ONLY_IN_FILE" not in os.environ


def test_missing_default_env_file_is_fine(clean_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert isinstance(load_env(), dict)


def test_missing_explicit_env_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="env file not found"):
        load_env(tmp_path / "nope.env", required=True)


# --- command line --------------------------------------------------------------------------

def test_cli_prints_default_weights(clean_env, tmp_path, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text("")
    assert main(["--input", str(DATA), "--env-file", str(env_file)]) == 0
    out = capsys.readouterr().out
    assert "weights: industry 0.4, size 0.25, source 0.2, recency 0.15" in out
    assert "(env)" not in out


def test_cli_applies_env_file_and_marks_overrides(clean_env, tmp_path, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text("WEIGHT_INDUSTRY=0.30\nWEIGHT_SIZE=0.35\n")
    assert main(["--input", str(DATA), "--env-file", str(env_file)]) == 0
    assert "weights: industry 0.3 (env), size 0.35 (env), source 0.2, recency 0.15" in capsys.readouterr().out


def test_cli_real_env_variable_beats_env_file(clean_env, tmp_path, monkeypatch, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text("WEIGHT_INDUSTRY=0.30\nWEIGHT_SIZE=0.35\n")
    monkeypatch.setenv("WEIGHT_INDUSTRY", "0.40")
    monkeypatch.setenv("WEIGHT_SIZE", "0.25")
    assert main(["--input", str(DATA), "--env-file", str(env_file)]) == 0
    assert "weights: industry 0.4 (env), size 0.25 (env)" in capsys.readouterr().out


def test_cli_fails_when_weights_do_not_add_up(clean_env, tmp_path, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text("WEIGHT_INDUSTRY=0.90\n")
    assert main(["--input", str(DATA), "--env-file", str(env_file)]) == 1
    assert "add up to 1" in capsys.readouterr().err


def test_cli_fails_on_missing_explicit_env_file(clean_env, tmp_path, capsys):
    assert main(["--input", str(DATA), "--env-file", str(tmp_path / "missing.env")]) == 1
    assert "env file not found" in capsys.readouterr().err


def test_cli_fails_on_misspelled_weight_variable(clean_env, tmp_path, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text("WEIGHT_RECENSY=0.15\n")
    assert main(["--input", str(DATA), "--env-file", str(env_file)]) == 1
    assert "unknown weight variable WEIGHT_RECENSY" in capsys.readouterr().err
