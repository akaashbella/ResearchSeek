import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from researchseek.cli import main


def test_console_help():
    executable = Path(sysconfig.get_path("scripts")) / "researchseek"
    result = subprocess.run([str(executable), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage: researchseek" in result.stdout
    for name in ("index", "search", "benchmark"):
        assert name in result.stdout
    assert result.stderr == ""


def test_module_help():
    result = subprocess.run(
        [sys.executable, "-m", "researchseek", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage: researchseek" in result.stdout


@pytest.mark.parametrize("command", ["index", "search", "benchmark"])
def test_command_help(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([command, "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    expected = {"index": "--download-model", "search": "--top-k", "benchmark": "Not implemented yet"}
    assert expected[command] in output


@pytest.mark.parametrize("arguments", [[], ["unknown"], ["--unknown"], ["index", "--unknown"]])
def test_invalid_usage(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["index", "search", "benchmark"])
def test_command_without_arguments(command, capsys):
    if command in {"index", "search"}:
        with pytest.raises(SystemExit) as error:
            main([command])
        assert error.value.code == 2
        assert ("directory" if command == "index" else "query") in capsys.readouterr().err
        return
    assert main([command]) == 1
    output = capsys.readouterr()
    assert "not implemented yet" in output.err
    assert output.out == ""
