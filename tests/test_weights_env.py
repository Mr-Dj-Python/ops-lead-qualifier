import os
from pathlib import Path

import pytest

from main import main
from pipeline.config import ConfigError, load_config, load_env

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
DATA = ROOT / "data" / "leads_training.csv"
WEIGHT_VARS = ["WEIGHT_INDUSTRY", "WEIGHT_SIZE", "WEIGHT_SOURCE", "WEIGHT_RECENCY"]
DEFAULT_ENV = {
    "WEIGHT_INDUSTRY": "0.40",
    "WEIGHT_SIZE": "0.25",
    "WEIGHT_SOURCE": "0.20",
    "WEIGHT_RECENCY": "0.15",
}


@pytest.fixture
def clean_env(monkeypatch):
    for name in WEIGHT_VARS:
        monkeypatch.delenv(name, raising=False)


def write_env(tmp_path, **values):
    path = tmp_path / ".env"
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    return path


# --- config.yaml refers to the .env values -------------------------------------------------

def test_config_yaml_holds_placeholders_not_numbers():
    text = CONFIG.read_text()
    for name in WEIGHT_VARS:
        assert "${" + name + "}" in text


def test_weights_come_from_the_env_values():
    config = load_config(CONFIG, env=DEFAULT_ENV)
    assert config["weights"] == {"industry": 0.40, "size": 0.25, "source": 0.20, "recency": 0.15}


def test_changing_the_env_values_changes_the_weights():
    env = {"WEIGHT_INDUSTRY": "0.5", "WEIGHT_SIZE": "0.2", "WEIGHT_SOURCE": "0.2", "WEIGHT_RECENCY": "0.1"}
    assert load_config(CONFIG, env=env)["weights"] == {"industry": 0.5, "size": 0.2, "source": 0.2, "recency": 0.1}


def test_zero_weight_is_allowed():
    env = {**DEFAULT_ENV, "WEIGHT_RECENCY": "0", "WEIGHT_INDUSTRY": "0.55"}
    assert load_config(CONFIG, env=env)["weights"]["recency"] == 0


def test_whitespace_around_values_is_ignored():
    assert load_config(CONFIG, env={**DEFAULT_ENV, "WEIGHT_SIZE": " 0.25 "})["weights"]["size"] == 0.25


@pytest.mark.parametrize("missing", WEIGHT_VARS)
def test_missing_variable_is_named_in_the_error(missing):
    env = {key: value for key, value in DEFAULT_ENV.items() if key != missing}
    with pytest.raises(ConfigError, match=rf"weights\.\w+ refers to \$\{{{missing}\}}.*add {missing} to \.env"):
        load_config(CONFIG, env=env)


def test_misspelled_variable_in_env_shows_up_as_a_missing_one():
    env = {key: value for key, value in DEFAULT_ENV.items() if key != "WEIGHT_RECENCY"} | {"WEIGHT_RECENSY": "0.15"}
    with pytest.raises(ConfigError, match="WEIGHT_RECENCY"):
        load_config(CONFIG, env=env)


@pytest.mark.parametrize("value", ["lots", "", "0.4.0", "nan", "inf", "-0.1"])
def test_invalid_weight_values_are_rejected(value):
    with pytest.raises(ConfigError, match=r"weights\.size must be a number"):
        load_config(CONFIG, env={**DEFAULT_ENV, "WEIGHT_SIZE": value})


def test_weights_must_add_up_to_one_and_the_error_shows_them():
    with pytest.raises(ConfigError, match=r"add up to 1, got 1\.1.*industry 0\.5"):
        load_config(CONFIG, env={**DEFAULT_ENV, "WEIGHT_INDUSTRY": "0.5"})


def test_literal_numbers_in_yaml_still_work(tmp_path):
    text = CONFIG.read_text()
    for name, value in DEFAULT_ENV.items():
        text = text.replace("${" + name + "}", value)
    path = tmp_path / "config.yaml"
    path.write_text(text)
    assert load_config(path, env={})["weights"]["industry"] == 0.4


def test_placeholders_work_for_other_settings_too(tmp_path):
    text = CONFIG.read_text().replace('reference_date: "2024-01-21"', 'reference_date: "${REF_DATE}"')
    path = tmp_path / "config.yaml"
    path.write_text(text)
    assert load_config(path, env={**DEFAULT_ENV, "REF_DATE": "2024-02-01"})["reference_date"] == "2024-02-01"


# --- load_env ------------------------------------------------------------------------------

def test_load_env_reads_the_file(clean_env, tmp_path):
    path = write_env(tmp_path, WEIGHT_SIZE="0.30", OTHER="1")
    env = load_env(path)
    assert env["WEIGHT_SIZE"] == "0.30" and env["OTHER"] == "1"


def test_real_environment_variable_beats_the_file(tmp_path, monkeypatch):
    path = write_env(tmp_path, WEIGHT_SIZE="0.30")
    monkeypatch.setenv("WEIGHT_SIZE", "0.25")
    assert load_env(path)["WEIGHT_SIZE"] == "0.25"


def test_load_env_does_not_modify_the_process_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("ONLY_IN_FILE", raising=False)
    path = write_env(tmp_path, ONLY_IN_FILE="1")
    assert load_env(path)["ONLY_IN_FILE"] == "1"
    assert "ONLY_IN_FILE" not in os.environ


def test_missing_default_env_file_is_not_an_error_by_itself(clean_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert isinstance(load_env(), dict)


def test_missing_explicit_env_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="env file not found"):
        load_env(tmp_path / "nope.env", required=True)


# --- command line --------------------------------------------------------------------------

def test_repo_env_file_provides_every_weight():
    from dotenv import dotenv_values

    values = dotenv_values(ROOT / ".env")
    assert {name: values.get(name) for name in WEIGHT_VARS} == DEFAULT_ENV


def test_cli_prints_the_weights_from_the_env_file(clean_env, tmp_path, capsys):
    path = write_env(tmp_path, **{**DEFAULT_ENV, "WEIGHT_INDUSTRY": "0.30", "WEIGHT_SIZE": "0.35"})
    assert main(["--input", str(DATA), "--env-file", str(path)]) == 0
    assert "weights: industry 0.3, size 0.35, source 0.2, recency 0.15" in capsys.readouterr().out


def test_cli_real_env_variable_beats_env_file(clean_env, tmp_path, monkeypatch, capsys):
    path = write_env(tmp_path, **{**DEFAULT_ENV, "WEIGHT_INDUSTRY": "0.30", "WEIGHT_SIZE": "0.35"})
    monkeypatch.setenv("WEIGHT_INDUSTRY", "0.40")
    monkeypatch.setenv("WEIGHT_SIZE", "0.25")
    assert main(["--input", str(DATA), "--env-file", str(path)]) == 0
    assert "weights: industry 0.4, size 0.25, source 0.2, recency 0.15" in capsys.readouterr().out


def test_cli_fails_when_weights_do_not_add_up(clean_env, tmp_path, capsys):
    path = write_env(tmp_path, **{**DEFAULT_ENV, "WEIGHT_INDUSTRY": "0.90"})
    assert main(["--input", str(DATA), "--env-file", str(path)]) == 1
    assert "add up to 1" in capsys.readouterr().err


def test_cli_fails_when_a_weight_is_missing(clean_env, tmp_path, capsys):
    path = write_env(tmp_path, WEIGHT_INDUSTRY="0.4", WEIGHT_SIZE="0.25", WEIGHT_SOURCE="0.2")
    assert main(["--input", str(DATA), "--env-file", str(path)]) == 1
    assert "add WEIGHT_RECENCY to .env" in capsys.readouterr().err


def test_cli_fails_without_any_env_file_or_variables(clean_env, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no .env here
    assert main(["--input", str(DATA), "--config", str(CONFIG)]) == 1
    assert "WEIGHT_INDUSTRY" in capsys.readouterr().err


def test_cli_fails_on_missing_explicit_env_file(clean_env, tmp_path, capsys):
    assert main(["--input", str(DATA), "--env-file", str(tmp_path / "missing.env")]) == 1
    assert "env file not found" in capsys.readouterr().err
