import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_CONFIG_KEYS = [
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
    "START_DATE",
    "END_DATE",
    "TARGET_DAYS",
    "WEIBO_INPUT_DIR",
    "WEIBO_FILENAME_PATTERN",
    "TWITTER_INPUT_DIR",
    "TWITTER_FILENAME_PATTERN",
    "OUTPUT_DIR",
    "MAX_WORKERS",
    "MAX_RETRIES",
    "REQUEST_TIMEOUT",
]


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_example_has_all_required_keys():
    config_example_path = REPO_ROOT / "config.example.py"
    module = _load_module(config_example_path, "config_example")
    for key in REQUIRED_CONFIG_KEYS:
        assert hasattr(module, key), f"config.example.py missing {key}"


def test_gitignore_excludes_config_py():
    gitignore_text = (REPO_ROOT / ".gitignore").read_text()
    assert "config.py" in gitignore_text.splitlines()


def test_gitignore_excludes_output_dir():
    gitignore_text = (REPO_ROOT / ".gitignore").read_text()
    assert "output/" in gitignore_text.splitlines()
