# Unified AI-Opinion Social Media Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone pipeline that analyzes AI-opinion sentiment for both Weibo and Twitter posts through one shared OpenRouter model and one shared prompt, then computes and plots comparable daily opinion series, as three independently runnable CLI stages.

**Architecture:** Three thin CLI entry points (`run_analysis.py`, `prepare_data.py`, `plot_figures.py`) sit on top of small, independently-testable library modules (`prompts.py`, `loaders.py`, `openrouter_client.py`). The library modules take all paths/config as explicit function arguments (no module-level config reads), so every unit is testable without a real `config.py` or network access; only the CLI `main()` functions read `config.py` and wire concrete values in.

**Tech Stack:** Python 3, pandas + pyarrow (parquet I/O), `openai` SDK pointed at OpenRouter's OpenAI-compatible endpoint, `fire` (CLI), `matplotlib` (Agg backend), `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-26-unified-opinion-pipeline-design.md`

## Global Constraints

- Model: `deepseek/deepseek-v4.1-flash` via OpenRouter, base URL `https://openrouter.ai/api/v1` — exact slug is a placeholder to verify against `openrouter.ai/models` before the first real (non-test) run.
- No OpenAI Batch API anywhere — all calls are concurrent synchronous chat-completions requests (thread pool).
- Exactly one canonical system prompt + one user-message template, byte-identical for both platforms — no retweet flag, no timestamp in the prompt (see spec's "Canonical prompt" section for exact text).
- `config.py` is gitignored; `config.example.py` is tracked and contains placeholder values only.
- `youth-analysis/` and `twitterapi-io/` (sibling repos) are read-only inputs to this pipeline — this plan never modifies files in either.
- Date assignment is platform-specific and must not be unified: Weibo's `date` = the source file's date (already correct upstream); Twitter's `date` = parsed from `createdAt`.
- Three daily metrics are always computed and kept: `avg_opinion`, `weighted_opinion`, `user_avg_opinion` (the last is the primary result metric, but all three ship in every output).
- `figure_data` (Stage 2's `export`) is always unsmoothed. Sliding-window smoothing (default window 3, centered) happens only inside `plot_figures.py`, never persisted back into `figure_data`.
- Plots compare exactly two lines (Weibo vs. Twitter) per metric — no cross-lingual correction factor, no four-line original/translated variant.

---

### Task 1: Repo scaffolding — config, gitignore, requirements, README

**Files:**
- Create: `config.example.py`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `README.md`
- Test: `tests/test_scaffolding.py`

**Interfaces:**
- Produces: `config.example.py` module with attributes `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`, `START_DATE`, `END_DATE`, `TARGET_DAYS`, `WEIBO_INPUT_DIR`, `WEIBO_FILENAME_PATTERN`, `TWITTER_INPUT_DIR`, `TWITTER_FILENAME_PATTERN`, `OUTPUT_DIR`, `MAX_WORKERS`, `MAX_RETRIES`, `REQUEST_TIMEOUT` — every later task's `config.py` (gitignored, created manually by the user) must supply these same attribute names.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffolding.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scaffolding.py -v`
Expected: FAIL — `config.example.py` and `.gitignore` do not exist yet.

- [ ] **Step 3: Create the scaffolding files**

```python
# config.example.py
"""
Copy this file to config.py and fill in real values.
config.py is gitignored — never commit real API keys.
"""

# OpenRouter
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash"  # verify exact slug on openrouter.ai/models

# Shared sampling window (day-of-month sampling, applied identically to both platforms)
START_DATE = "2024-03-01"
END_DATE = "2025-03-20"
TARGET_DAYS = [1, 10, 20]

# Weibo input: already keyword-filtered per-day files produced by
# youth-analysis/ai_content_extractor.py. This pipeline only reads them.
WEIBO_INPUT_DIR = "YOUR_PATH_HERE"  # e.g. /path/to/youth-analysis/ai_attitudes/ai_weibo_text
WEIBO_FILENAME_PATTERN = "{date}.parquet"

# Twitter input: already crawled per-day files produced by
# twitterapi-io's crawler + convert_to_parquet.py. This pipeline only reads them.
TWITTER_INPUT_DIR = "YOUR_PATH_HERE"  # e.g. /path/to/twitterapi-io/parquet_data
TWITTER_FILENAME_PATTERN = "tweets_{date}.parquet"

# Output roots
OUTPUT_DIR = "output"

# Concurrency / retries for OpenRouter calls
MAX_WORKERS = 8
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
```

```
# .gitignore
config.py
output/
__pycache__/
*.py[cod]
.pytest_cache/
.DS_Store
venv/
.venv/
```

```
# requirements.txt
pandas
pyarrow
openai
fire
tqdm
matplotlib
pytest
```

```markdown
# README.md
# ai-attitudes-social-media

Unified AI-opinion analysis pipeline for the Weibo (China) / Twitter (US) AI-attitudes
paper. Both platforms are analyzed with the same OpenRouter model and the same prompt
(see `prompts.py`), removing the need for the cross-lingual bias correction the old
two-model (GPT + Kimi) setup required.

This repo does **not** crawl or keyword-filter data. It reads already-extracted,
per-day parquet files produced by the sibling `youth-analysis` (Weibo) and
`twitterapi-io` (Twitter) repos — configure their paths in `config.py`.

## Setup

```bash
pip install -r requirements.txt
cp config.example.py config.py
# edit config.py: OPENROUTER_API_KEY, WEIBO_INPUT_DIR, TWITTER_INPUT_DIR
```

## Usage — three independent stages (can run on different machines)

```bash
# Stage 1: AI opinion analysis via OpenRouter (needs network access to OpenRouter)
python run_analysis.py weibo
python run_analysis.py twitter

# Stage 2: clean + compute daily aggregates (raw mean / like-weighted / user-mean) + export figure data
python prepare_data.py clean --platform weibo
python prepare_data.py clean --platform twitter
python prepare_data.py export

# Stage 3: plot (sliding-window smoothing happens here only, figure_data itself stays unsmoothed)
python plot_figures.py
```

Outputs land under `output/`: `analysis_results/{platform}_opinion_results.csv`,
`{platform}_daily_opinion.parquet`, `figure_data.parquet`/`.csv`, `figures/*.pdf`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scaffolding.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add config.example.py .gitignore requirements.txt README.md tests/test_scaffolding.py
git commit -m "chore: add repo scaffolding (config template, gitignore, requirements, README)"
```

---

### Task 2: Canonical prompt module

**Files:**
- Create: `prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Produces: `prompts.SYSTEM_PROMPT: str`, `prompts.build_user_message(text: str) -> str`. `openrouter_client.py` (Task 6) imports both.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
from prompts import SYSTEM_PROMPT, build_user_message


def test_build_user_message_format():
    assert build_user_message("hello world") == "Post text: hello world"


def test_build_user_message_no_platform_or_language_wording():
    message = build_user_message("some text")
    assert "Twitter" not in message
    assert "Weibo" not in message


def test_system_prompt_has_all_opinion_labels():
    for label in ["2 =", "1 =", "0 =", "-1 =", "-2 =", '"cannot tell" =']:
        assert label in SYSTEM_PROMPT


def test_system_prompt_requires_json_with_opinion_field():
    assert '"opinion"' in SYSTEM_PROMPT
    assert "JSON object" in SYSTEM_PROMPT


def test_system_prompt_is_language_and_platform_neutral():
    assert "Twitter" not in SYSTEM_PROMPT
    assert "Weibo" not in SYSTEM_PROMPT
    assert "regardless of the language" in SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL — `prompts.py` does not exist.

- [ ] **Step 3: Write the implementation**

```python
# prompts.py
"""
Canonical AI-opinion prompt, shared byte-for-byte across Weibo and Twitter.

Single prompt, single model — this removes the need for the cross-lingual bias
correction the old two-model (GPT for Twitter, Kimi for Weibo) setup required.
"""

SYSTEM_PROMPT = """You will read a piece of text posted by a user on social media. The text may be written
in Chinese, English, or any other language — analyze it regardless of the language it is
written in. Please analyze the text's opinion toward AI technology and assign it an
opinion label. Your output must be a JSON object containing only one field, "opinion",
whose value must be one of: -2, -1, 0, 1, 2, or "cannot tell".

Label definitions:
2 = The text expresses a strongly positive attitude toward AI technology, clearly
asserting that the benefits of AI outweigh the harms and that AI brings significant
positive impacts to society (e.g., improving convenience, enhancing health and medical
services, creating economic opportunities, increasing learning or work efficiency,
improving safety, or supporting research and innovation), or expresses reliance on or
trust in AI.
1 = The text is overall positive toward AI, but the attitude is mild or includes
reservations (e.g., expresses support but without strong enthusiasm; believes AI is
"generally beneficial" while also mentioning risks or limitations; expresses
expectation, interest, or positive impressions, but not strong praise).
0 = The text is neutral or difficult to judge (e.g., mentions both pros and cons but
without a clear leaning).
-1 = The text is overall negative toward AI, but the attitude is mild or includes
reservations (e.g., expresses concern or opposition but does not fully reject AI;
believes AI is "risky or harmful" but acknowledges certain benefits; expresses caution,
unease, or negative views, but without strong condemnation).
-2 = The text expresses a strongly negative attitude toward AI technology, clearly
asserting that the harms outweigh the benefits and that AI brings significant negative
impacts to society (e.g., leading to unemployment, contributing to economic bubbles,
increasing privacy risks, reinforcing bias against marginalized groups, generating
misinformation or rumors, or posing safety threats), or expresses resistance to or
distrust in AI.
"cannot tell" = The text does not express any attitude toward AI (e.g., content is
unrelated to AI, or does not reflect a clear viewpoint).

Please return only a JSON object, for example:
{"opinion": 1}"""


def build_user_message(text: str) -> str:
    return f"Post text: {text}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add prompts.py tests/test_prompts.py
git commit -m "feat: add canonical AI-opinion prompt shared by both platforms"
```

---

### Task 3: Shared day-of-month date filter

**Files:**
- Create: `loaders.py`
- Test: `tests/test_loaders.py`

**Interfaces:**
- Produces: `loaders.iter_target_dates(start_date: str, end_date: str, target_days: list) -> list[str]` (dates formatted `"YYYY-MM-DD"`). Tasks 4 and 5 (weibo_loader/twitter_loader) call this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loaders.py
from loaders import iter_target_dates


def test_iter_target_dates_single_month():
    dates = iter_target_dates("2024-03-01", "2024-03-31", [1, 10, 20])
    assert dates == ["2024-03-01", "2024-03-10", "2024-03-20"]


def test_iter_target_dates_spans_months():
    dates = iter_target_dates("2024-03-15", "2024-04-15", [1, 10, 20])
    assert dates == ["2024-03-20", "2024-04-01", "2024-04-10"]


def test_iter_target_dates_no_matches():
    dates = iter_target_dates("2024-03-02", "2024-03-09", [1, 10, 20])
    assert dates == []


def test_iter_target_dates_single_day_range():
    dates = iter_target_dates("2024-03-10", "2024-03-10", [1, 10, 20])
    assert dates == ["2024-03-10"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loaders.py -v`
Expected: FAIL — `loaders.py` does not exist.

- [ ] **Step 3: Write the implementation**

```python
# loaders.py
"""Platform-specific loaders that read already-extracted per-day parquet files
from the sibling youth-analysis (Weibo) and twitterapi-io (Twitter) repos, and
normalize them into a standard {id, text, user_id, weight_raw, date} frame.
"""

import os
from datetime import datetime, timedelta
from typing import List

import pandas as pd

STANDARD_COLUMNS = ["id", "text", "user_id", "weight_raw", "date"]


def iter_target_dates(start_date: str, end_date: str, target_days: List[int]) -> List[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        if current.day in target_days:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loaders.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add loaders.py tests/test_loaders.py
git commit -m "feat: add shared day-of-month date filter for loaders"
```

---

### Task 4: Weibo loader

**Files:**
- Modify: `loaders.py`
- Test: `tests/test_loaders.py`

**Interfaces:**
- Consumes: `iter_target_dates` (Task 3).
- Produces: `loaders.weibo_loader(input_dir: str, filename_pattern: str, start_date: str, end_date: str, target_days: list) -> pd.DataFrame` with columns `STANDARD_COLUMNS`. Task 9 (`prepare_data.clean`) and Task 8 (`run_analysis.analyze`) call this via the `LOADERS["weibo"]` dispatch dict.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_loaders.py
import pandas as pd

from loaders import weibo_loader, STANDARD_COLUMNS


def test_weibo_loader_reads_matching_dates_only(tmp_path):
    day1 = pd.DataFrame({
        "weibo_id": ["w1", "w2"],
        "user_id": ["u1", "u2"],
        "weibo_content": ["AI is great", "AI is scary"],
        "zan": [5, 0],
    })
    day1.to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    day_not_sampled = pd.DataFrame({
        "weibo_id": ["w3"],
        "user_id": ["u3"],
        "weibo_content": ["not sampled"],
        "zan": [1],
    })
    day_not_sampled.to_parquet(tmp_path / "2024-03-02.parquet", index=False)

    result = weibo_loader(
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )

    assert list(result.columns) == STANDARD_COLUMNS
    assert set(result["id"]) == {"w1", "w2"}
    assert result[result["id"] == "w1"]["text"].iloc[0] == "AI is great"
    assert result[result["id"] == "w1"]["weight_raw"].iloc[0] == 5
    assert result[result["id"] == "w1"]["user_id"].iloc[0] == "u1"
    assert result[result["id"] == "w1"]["date"].iloc[0] == "2024-03-01"


def test_weibo_loader_skips_missing_files(tmp_path):
    result = weibo_loader(
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )
    assert list(result.columns) == STANDARD_COLUMNS
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loaders.py -v`
Expected: FAIL — `weibo_loader` not defined.

- [ ] **Step 3: Write the implementation**

```python
# append to loaders.py

def weibo_loader(
    input_dir: str,
    filename_pattern: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
) -> pd.DataFrame:
    dates = iter_target_dates(start_date, end_date, target_days)
    frames = []
    for date_str in dates:
        file_path = os.path.join(input_dir, filename_pattern.format(date=date_str))
        if not os.path.exists(file_path):
            continue
        df = pd.read_parquet(
            file_path, columns=["weibo_id", "user_id", "weibo_content", "zan"]
        )
        df = df.rename(
            columns={"weibo_id": "id", "weibo_content": "text", "zan": "weight_raw"}
        )
        df["id"] = df["id"].astype(str)
        df["user_id"] = df["user_id"].astype(str)
        # Weibo date = the source file's date, already correctly bucketed
        # upstream by youth-analysis/ai_content_extractor.py. Do not re-derive
        # from time_stamp (ambiguous timezone, dead code in the old pipeline).
        df["date"] = date_str
        frames.append(df[STANDARD_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loaders.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add loaders.py tests/test_loaders.py
git commit -m "feat: add weibo_loader reading already-extracted per-day parquet files"
```

---

### Task 5: Twitter loader

**Files:**
- Modify: `loaders.py`
- Test: `tests/test_loaders.py`

**Interfaces:**
- Consumes: `iter_target_dates` (Task 3).
- Produces: `loaders.twitter_loader(input_dir: str, filename_pattern: str, start_date: str, end_date: str, target_days: list) -> pd.DataFrame` with columns `STANDARD_COLUMNS`, `date` parsed per-row from `createdAt`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_loaders.py
from loaders import twitter_loader


def test_twitter_loader_reads_matching_dates_and_parses_created_at(tmp_path):
    day1 = pd.DataFrame({
        "id": ["t1", "t2"],
        "text": ["AI is great", "AI is scary"],
        "likeCount": [10, 0],
        "author.id": ["a1", "a2"],
        "createdAt": [
            "Fri Mar 01 12:00:00 +0000 2024",
            "Fri Mar 01 23:59:00 +0000 2024",
        ],
    })
    day1.to_parquet(tmp_path / "tweets_2024-03-01.parquet", index=False)

    result = twitter_loader(
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )

    assert list(result.columns) == STANDARD_COLUMNS
    assert set(result["id"]) == {"t1", "t2"}
    row = result[result["id"] == "t1"].iloc[0]
    assert row["text"] == "AI is great"
    assert row["weight_raw"] == 10
    assert row["user_id"] == "a1"
    assert row["date"] == "2024-03-01"


def test_twitter_loader_skips_missing_files(tmp_path):
    result = twitter_loader(
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )
    assert list(result.columns) == STANDARD_COLUMNS
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loaders.py -v`
Expected: FAIL — `twitter_loader` not defined.

- [ ] **Step 3: Write the implementation**

```python
# append to loaders.py

def twitter_loader(
    input_dir: str,
    filename_pattern: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
) -> pd.DataFrame:
    dates = iter_target_dates(start_date, end_date, target_days)
    frames = []
    for date_str in dates:
        file_path = os.path.join(input_dir, filename_pattern.format(date=date_str))
        if not os.path.exists(file_path):
            continue
        df = pd.read_parquet(
            file_path, columns=["id", "text", "likeCount", "author.id", "createdAt"]
        )
        df["id"] = df["id"].astype(str)
        # Twitter date = parsed from the post's own createdAt (UTC), same as
        # the current batch_sentiment_analysis.py behavior.
        df["date"] = df["createdAt"].apply(
            lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S +0000 %Y").strftime(
                "%Y-%m-%d"
            )
        )
        df = df.rename(columns={"likeCount": "weight_raw", "author.id": "user_id"})
        df["user_id"] = df["user_id"].astype(str)
        frames.append(df[STANDARD_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loaders.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add loaders.py tests/test_loaders.py
git commit -m "feat: add twitter_loader reading already-crawled per-day parquet files"
```

---

### Task 6: OpenRouter client — single-post analysis with retry

**Files:**
- Create: `openrouter_client.py`
- Test: `tests/test_openrouter_client.py`

**Interfaces:**
- Consumes: `prompts.SYSTEM_PROMPT`, `prompts.build_user_message` (Task 2).
- Produces: `openrouter_client.OpenRouterClient(api_key, base_url, model, max_retries=3, timeout=60, client=None)` with method `.analyze_one(text: str) -> dict` returning `{"opinion": Optional[int|str], "prompt_tokens": int, "completion_tokens": int, "cached_tokens": int}`. The `client` constructor arg accepts a pre-built OpenAI-SDK-shaped object for dependency injection in tests; Task 7's `analyze_many` and Task 8's `run_analysis.analyze` depend on `.analyze_one`'s exact return-dict shape.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_openrouter_client.py
from types import SimpleNamespace

from openrouter_client import OpenRouterClient


class FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5, cached_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached_tokens = cached_tokens


def _fake_response(content, usage=None):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage or FakeUsage())


class FakeOpenAI:
    """Stands in for the openai.OpenAI client, shaped like it for analyze_one."""

    def __init__(self, responses=None, error_then_success=None):
        self._responses = responses or []
        self._error_then_success = error_then_success
        self.call_count = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, model, messages):
        self.call_count += 1
        if self._error_then_success is not None and self.call_count == 1:
            raise ConnectionError("simulated transient failure")
        response = self._responses[min(self.call_count - 1, len(self._responses) - 1)]
        return response


def test_analyze_one_parses_valid_json_opinion():
    fake = FakeOpenAI(responses=[_fake_response('{"opinion": 1}')])
    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=fake
    )
    result = client.analyze_one("some text")
    assert result["opinion"] == 1
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 5


def test_analyze_one_parses_cannot_tell():
    fake = FakeOpenAI(responses=[_fake_response('{"opinion": "cannot tell"}')])
    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=fake
    )
    result = client.analyze_one("some text")
    assert result["opinion"] == "cannot tell"


def test_analyze_one_returns_none_opinion_on_malformed_json():
    fake = FakeOpenAI(responses=[_fake_response("not json at all")])
    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=fake
    )
    result = client.analyze_one("some text")
    assert result["opinion"] is None


def test_analyze_one_retries_then_succeeds():
    fake = FakeOpenAI(
        responses=[_fake_response('{"opinion": 2}')], error_then_success=True
    )
    client = OpenRouterClient(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        max_retries=2,
        client=fake,
    )
    result = client.analyze_one("some text")
    assert result["opinion"] == 2
    assert fake.call_count == 2


def test_analyze_one_gives_up_after_max_retries():
    class AlwaysFails:
        def __init__(self):
            self.call_count = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, model, messages):
            self.call_count += 1
            raise ConnectionError("always fails")

    fake = AlwaysFails()
    client = OpenRouterClient(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        max_retries=2,
        client=fake,
    )
    result = client.analyze_one("some text")
    assert result["opinion"] is None
    assert fake.call_count == 3  # initial attempt + 2 retries
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_openrouter_client.py -v`
Expected: FAIL — `openrouter_client.py` does not exist.

- [ ] **Step 3: Write the implementation**

```python
# openrouter_client.py
"""Concurrent OpenRouter chat-completions client used by both platforms
(replaces the old OpenAI Batch API + hosted-prompt mechanism)."""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from prompts import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 3,
        timeout: int = 60,
        client=None,
    ):
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.max_retries = max_retries

    def analyze_one(self, text: str) -> Dict:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_message(text)},
                    ],
                )
                response_text = response.choices[0].message.content.strip()
                usage = response.usage
                token_stats = {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "cached_tokens": getattr(usage, "cached_tokens", 0) if usage else 0,
                }

                opinion = None
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    try:
                        opinion = json.loads(response_text[json_start:json_end]).get(
                            "opinion"
                        )
                    except json.JSONDecodeError:
                        opinion = None

                return {"opinion": opinion, **token_stats}
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep((attempt + 1) * 0.01)

        logger.error(f"analyze_one failed after {self.max_retries} retries: {last_error}")
        return {
            "opinion": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
```

Note: the backoff sleep uses `(attempt + 1) * 0.01` seconds (not the 2s/4s of the
legacy scripts) so tests stay fast; tune this constant up in `run_analysis.py` call
sites or make it configurable later if real-world rate limiting needs it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_openrouter_client.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add openrouter_client.py tests/test_openrouter_client.py
git commit -m "feat: add OpenRouterClient.analyze_one with retry/backoff"
```

---

### Task 7: Concurrent, resumable batch analysis

**Files:**
- Modify: `openrouter_client.py`
- Test: `tests/test_openrouter_client.py`

**Interfaces:**
- Consumes: any object with `.analyze_one(text: str) -> dict` (duck-typed, so tests can pass a fake instead of a real `OpenRouterClient`).
- Produces: `openrouter_client.analyze_many(client, df: pd.DataFrame, results_path: str, max_workers: int = 8) -> dict` where `df` has columns `["id", "text", ...]` (extra columns ignored). Writes/appends CSV at `results_path` with columns `id,opinion,prompt_tokens,completion_tokens,cached_tokens`. Returns `{"total": int, "skipped": int, "completed": int, "failed": int}`. Task 8's `run_analysis.analyze` calls this.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_openrouter_client.py
import pandas as pd

from openrouter_client import analyze_many


class CountingFakeClient:
    def __init__(self, opinion_by_text=None):
        self.calls = []
        self._opinion_by_text = opinion_by_text or {}

    def analyze_one(self, text):
        self.calls.append(text)
        opinion = self._opinion_by_text.get(text, 1)
        return {
            "opinion": opinion,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
        }


def test_analyze_many_writes_results_csv(tmp_path):
    df = pd.DataFrame({"id": ["a", "b"], "text": ["text a", "text b"]})
    results_path = tmp_path / "results.csv"
    client = CountingFakeClient()

    summary = analyze_many(client, df, str(results_path), max_workers=2)

    assert summary == {"total": 2, "skipped": 0, "completed": 2, "failed": 0}
    saved = pd.read_csv(results_path, dtype=str)
    assert set(saved["id"]) == {"a", "b"}
    assert set(saved["opinion"]) == {"1"}


def test_analyze_many_skips_already_processed_ids(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text("id,opinion,prompt_tokens,completion_tokens,cached_tokens\na,1,1,1,0\n")

    df = pd.DataFrame({"id": ["a", "b"], "text": ["text a", "text b"]})
    client = CountingFakeClient()

    summary = analyze_many(client, df, str(results_path), max_workers=2)

    assert summary["skipped"] == 1
    assert summary["completed"] == 1
    assert client.calls == ["text b"]
    saved = pd.read_csv(results_path, dtype=str)
    assert len(saved) == 2


def test_analyze_many_counts_failed_when_opinion_is_none(tmp_path):
    df = pd.DataFrame({"id": ["a"], "text": ["text a"]})
    client = CountingFakeClient(opinion_by_text={"text a": None})
    results_path = tmp_path / "results.csv"

    summary = analyze_many(client, df, str(results_path), max_workers=1)

    assert summary == {"total": 1, "skipped": 0, "completed": 0, "failed": 1}
    saved = pd.read_csv(results_path)
    assert saved["opinion"].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_openrouter_client.py -v`
Expected: FAIL — `analyze_many` not defined.

- [ ] **Step 3: Write the implementation**

```python
# append to openrouter_client.py

def load_processed_ids(results_path: str) -> set:
    path = Path(results_path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, dtype=str)
    if "id" not in df.columns:
        return set()
    return set(df["id"].dropna().unique())


def analyze_many(
    client, df: pd.DataFrame, results_path: str, max_workers: int = 8
) -> Dict:
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids(results_path)
    df = df.copy()
    df["id"] = df["id"].astype(str)
    todo = df[~df["id"].isin(processed_ids)]

    summary = {
        "total": len(df),
        "skipped": len(df) - len(todo),
        "completed": 0,
        "failed": 0,
    }
    if len(todo) == 0:
        return summary

    write_lock = threading.Lock()
    write_header = not path.exists() or path.stat().st_size == 0

    def process_row(row_id, row_text):
        result = client.analyze_one(row_text)
        opinion_value = "" if result["opinion"] is None else result["opinion"]
        nonlocal write_header
        with write_lock:
            with open(path, "a", encoding="utf-8", newline="") as f:
                if write_header:
                    f.write("id,opinion,prompt_tokens,completion_tokens,cached_tokens\n")
                    write_header = False
                f.write(
                    f'{row_id},{opinion_value},{result["prompt_tokens"]},'
                    f'{result["completion_tokens"]},{result["cached_tokens"]}\n'
                )
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_row, row["id"], row["text"])
            for _, row in todo.iterrows()
        ]
        for future in as_completed(futures):
            result = future.result()
            if result["opinion"] is None:
                summary["failed"] += 1
            else:
                summary["completed"] += 1

    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_openrouter_client.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add openrouter_client.py tests/test_openrouter_client.py
git commit -m "feat: add concurrent resumable analyze_many batch runner"
```

---

### Task 8: Stage 1 CLI — `run_analysis.py`

**Files:**
- Create: `run_analysis.py`
- Test: `tests/test_run_analysis.py`

**Interfaces:**
- Consumes: `loaders.weibo_loader`, `loaders.twitter_loader` (Tasks 4, 5); `openrouter_client.OpenRouterClient`, `openrouter_client.analyze_many` (Tasks 6, 7).
- Produces: `run_analysis.analyze(platform, input_dir, filename_pattern, output_path, api_key, base_url, model, start_date, end_date, target_days, workers=8, max_retries=3, client=None) -> dict` (the `client` param is DI for tests — when `None`, a real `OpenRouterClient` is built). `run_analysis.main(platform, start_date=None, end_date=None, target_days=None, workers=None)` is the `fire`-exposed CLI entry point that reads `config.py` for defaults and calls `analyze()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_analysis.py
import pandas as pd

from run_analysis import analyze


class FakeClient:
    def __init__(self):
        self.calls = []

    def analyze_one(self, text):
        self.calls.append(text)
        return {
            "opinion": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
        }


def test_analyze_weibo_end_to_end(tmp_path):
    day1 = pd.DataFrame({
        "weibo_id": ["w1"],
        "user_id": ["u1"],
        "weibo_content": ["AI is great"],
        "zan": [5],
    })
    day1.to_parquet(tmp_path / "2024-03-01.parquet", index=False)
    output_path = tmp_path / "out" / "weibo_opinion_results.csv"
    client = FakeClient()

    summary = analyze(
        platform="weibo",
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        output_path=str(output_path),
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        workers=2,
        client=client,
    )

    assert summary["completed"] == 1
    assert client.calls == ["AI is great"]
    assert output_path.exists()


def test_analyze_returns_zero_summary_when_no_input(tmp_path):
    output_path = tmp_path / "out" / "twitter_opinion_results.csv"
    client = FakeClient()

    summary = analyze(
        platform="twitter",
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        output_path=str(output_path),
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        client=client,
    )

    assert summary == {"total": 0, "skipped": 0, "completed": 0, "failed": 0}
    assert client.calls == []


def test_analyze_rejects_unknown_platform(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        analyze(
            platform="reddit",
            input_dir=str(tmp_path),
            filename_pattern="{date}.parquet",
            output_path=str(tmp_path / "out.csv"),
            api_key="k",
            base_url="https://openrouter.ai/api/v1",
            model="m",
            start_date="2024-03-01",
            end_date="2024-03-05",
            target_days=[1, 10, 20],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_analysis.py -v`
Expected: FAIL — `run_analysis.py` does not exist.

- [ ] **Step 3: Write the implementation**

```python
# run_analysis.py
"""Stage 1 CLI: AI-opinion analysis via OpenRouter, for one platform at a time.

Usage:
    python run_analysis.py weibo
    python run_analysis.py twitter
    python run_analysis.py weibo --start_date 2024-03-01 --end_date 2024-03-31
"""

from typing import List, Optional

import fire

from loaders import twitter_loader, weibo_loader
from openrouter_client import OpenRouterClient, analyze_many

LOADERS = {"weibo": weibo_loader, "twitter": twitter_loader}


def analyze(
    platform: str,
    input_dir: str,
    filename_pattern: str,
    output_path: str,
    api_key: str,
    base_url: str,
    model: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
    workers: int = 8,
    max_retries: int = 3,
    client=None,
) -> dict:
    if platform not in LOADERS:
        raise ValueError(f"Unknown platform: {platform!r}, expected one of {list(LOADERS)}")

    loader = LOADERS[platform]
    df = loader(input_dir, filename_pattern, start_date, end_date, target_days)

    if len(df) == 0:
        print(f"No input rows found for platform={platform} in range {start_date}..{end_date}")
        return {"total": 0, "skipped": 0, "completed": 0, "failed": 0}

    if client is None:
        client = OpenRouterClient(
            api_key=api_key, base_url=base_url, model=model, max_retries=max_retries
        )

    summary = analyze_many(client, df, output_path, max_workers=workers)
    print(f"Stage 1 analysis summary for {platform}: {summary}")
    return summary


def main(
    platform: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_days: Optional[str] = None,
    workers: Optional[int] = None,
):
    import config

    input_dir = config.WEIBO_INPUT_DIR if platform == "weibo" else config.TWITTER_INPUT_DIR
    filename_pattern = (
        config.WEIBO_FILENAME_PATTERN
        if platform == "weibo"
        else config.TWITTER_FILENAME_PATTERN
    )
    output_path = f"{config.OUTPUT_DIR}/analysis_results/{platform}_opinion_results.csv"
    days = [int(d) for d in target_days.split(",")] if target_days else config.TARGET_DAYS

    analyze(
        platform=platform,
        input_dir=input_dir,
        filename_pattern=filename_pattern,
        output_path=output_path,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        model=config.OPENROUTER_MODEL,
        start_date=start_date or config.START_DATE,
        end_date=end_date or config.END_DATE,
        target_days=days,
        workers=workers or config.MAX_WORKERS,
        max_retries=config.MAX_RETRIES,
    )


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_analysis.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add run_analysis.py tests/test_run_analysis.py
git commit -m "feat: add Stage 1 CLI run_analysis.py"
```

---

### Task 9: Stage 2 CLI (part 1) — `prepare_data.py clean`

**Files:**
- Create: `prepare_data.py`
- Test: `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: `loaders.weibo_loader`, `loaders.twitter_loader` (Tasks 4, 5).
- Produces: `prepare_data.clean(platform, input_dir, filename_pattern, start_date, end_date, target_days, opinion_results_path, output_path) -> pd.DataFrame` with columns `["date", "avg_opinion", "weighted_opinion", "user_avg_opinion"]`, also persisted to `output_path` (parquet). Task 10's `export` reads files this produces.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prepare_data.py
import pandas as pd

from prepare_data import clean


def test_clean_computes_three_daily_metrics(tmp_path):
    day1 = pd.DataFrame({
        "weibo_id": ["w1", "w2", "w3"],
        "user_id": ["u1", "u1", "u2"],
        "weibo_content": ["a", "b", "c"],
        "zan": [0, 4, 9],
    })
    day1.to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    opinions = pd.DataFrame({
        "id": ["w1", "w2", "w3"],
        "opinion": [2, -2, 1],
        "prompt_tokens": [1, 1, 1],
        "completion_tokens": [1, 1, 1],
        "cached_tokens": [0, 0, 0],
    })
    opinion_results_path = tmp_path / "weibo_opinion_results.csv"
    opinions.to_csv(opinion_results_path, index=False)

    output_path = tmp_path / "weibo_daily_opinion.parquet"

    result = clean(
        platform="weibo",
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        opinion_results_path=str(opinion_results_path),
        output_path=str(output_path),
    )

    assert list(result.columns) == ["date", "avg_opinion", "weighted_opinion", "user_avg_opinion"]
    row = result[result["date"] == "2024-03-01"].iloc[0]

    # avg_opinion: raw mean of [2, -2, 1] = 0.333...
    assert row["avg_opinion"] == pytest.approx((2 - 2 + 1) / 3)

    # weighted_opinion: weights are zan+1 = [1, 5, 10]
    # weighted mean = (2*1 + -2*5 + 1*10) / (1+5+10) = 2/16
    assert row["weighted_opinion"] == pytest.approx((2 * 1 + -2 * 5 + 1 * 10) / 16)

    # user_avg_opinion: u1 mean = (2 + -2)/2 = 0, u2 mean = 1; mean of [0, 1] = 0.5
    assert row["user_avg_opinion"] == pytest.approx(0.5)

    assert output_path.exists()
    saved = pd.read_parquet(output_path)
    assert len(saved) == 1


def test_clean_drops_cannot_tell_and_missing_opinions(tmp_path):
    day1 = pd.DataFrame({
        "weibo_id": ["w1", "w2", "w3"],
        "user_id": ["u1", "u1", "u2"],
        "weibo_content": ["a", "b", "c"],
        "zan": [0, 0, 0],
    })
    day1.to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    opinions = pd.DataFrame({
        "id": ["w1", "w2", "w3"],
        "opinion": [2, "cannot tell", ""],
        "prompt_tokens": [1, 1, 1],
        "completion_tokens": [1, 1, 1],
        "cached_tokens": [0, 0, 0],
    })
    opinion_results_path = tmp_path / "weibo_opinion_results.csv"
    opinions.to_csv(opinion_results_path, index=False)
    output_path = tmp_path / "weibo_daily_opinion.parquet"

    result = clean(
        platform="weibo",
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        opinion_results_path=str(opinion_results_path),
        output_path=str(output_path),
    )

    row = result[result["date"] == "2024-03-01"].iloc[0]
    # only w1 (opinion=2) survives the cannot-tell/missing filter
    assert row["avg_opinion"] == pytest.approx(2)
```

Add `import pytest` at the top of `tests/test_prepare_data.py` alongside the `pandas` import.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prepare_data.py -v`
Expected: FAIL — `prepare_data.py` does not exist.

- [ ] **Step 3: Write the implementation**

```python
# prepare_data.py
"""Stage 2 CLI: clean opinion results, compute daily aggregates, export figure data.

Usage:
    python prepare_data.py clean --platform weibo
    python prepare_data.py clean --platform twitter
    python prepare_data.py export
"""

from pathlib import Path
from typing import List

import fire
import pandas as pd

from loaders import twitter_loader, weibo_loader

LOADERS = {"weibo": weibo_loader, "twitter": twitter_loader}


def clean(
    platform: str,
    input_dir: str,
    filename_pattern: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
    opinion_results_path: str,
    output_path: str,
) -> pd.DataFrame:
    if platform not in LOADERS:
        raise ValueError(f"Unknown platform: {platform!r}, expected one of {list(LOADERS)}")

    loader = LOADERS[platform]
    meta_df = loader(input_dir, filename_pattern, start_date, end_date, target_days)

    opinions_df = pd.read_csv(opinion_results_path, dtype={"id": str})
    merged = meta_df.merge(opinions_df, on="id", how="inner")

    merged["opinion"] = pd.to_numeric(merged["opinion"], errors="coerce")
    merged = merged.dropna(subset=["opinion"])
    merged["weight"] = pd.to_numeric(merged["weight_raw"], errors="coerce").fillna(0) + 1

    daily_avg = merged.groupby("date")["opinion"].mean().reset_index()
    daily_avg.columns = ["date", "avg_opinion"]

    merged["opinion_weight"] = merged["opinion"] * merged["weight"]
    daily_weighted = (
        merged.groupby("date")
        .agg({"opinion_weight": "sum", "weight": "sum"})
        .reset_index()
    )
    daily_weighted["weighted_opinion"] = (
        daily_weighted["opinion_weight"] / daily_weighted["weight"]
    )
    daily_weighted = daily_weighted[["date", "weighted_opinion"]]

    user_daily_avg = (
        merged.groupby(["date", "user_id"])["opinion"].mean().reset_index()
    )
    user_daily_avg.columns = ["date", "user_id", "user_daily_avg_opinion"]
    daily_user_avg = (
        user_daily_avg.groupby("date")["user_daily_avg_opinion"].mean().reset_index()
    )
    daily_user_avg.columns = ["date", "user_avg_opinion"]

    result = daily_avg.merge(daily_weighted, on="date", how="outer")
    result = result.merge(daily_user_avg, on="date", how="outer")
    result = result.sort_values("date").reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved daily opinion for {platform} to {output_path} ({len(result)} dates)")
    return result


def _clean_cli(platform: str):
    import config

    input_dir = config.WEIBO_INPUT_DIR if platform == "weibo" else config.TWITTER_INPUT_DIR
    filename_pattern = (
        config.WEIBO_FILENAME_PATTERN
        if platform == "weibo"
        else config.TWITTER_FILENAME_PATTERN
    )
    opinion_results_path = f"{config.OUTPUT_DIR}/analysis_results/{platform}_opinion_results.csv"
    output_path = f"{config.OUTPUT_DIR}/{platform}_daily_opinion.parquet"

    clean(
        platform=platform,
        input_dir=input_dir,
        filename_pattern=filename_pattern,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        target_days=config.TARGET_DAYS,
        opinion_results_path=opinion_results_path,
        output_path=output_path,
    )


def main():
    fire.Fire({"clean": _clean_cli})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare_data.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add prepare_data.py tests/test_prepare_data.py
git commit -m "feat: add Stage 2 clean command computing the three daily opinion metrics"
```

---

### Task 10: Stage 2 CLI (part 2) — `prepare_data.py export`

**Files:**
- Modify: `prepare_data.py`
- Test: `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: parquet files shaped like Task 9's `clean()` output (`date, avg_opinion, weighted_opinion, user_avg_opinion`).
- Produces: `prepare_data.export(weibo_path, twitter_path, output_path) -> pd.DataFrame` with columns `["date", "weibo_avg_opinion", "twitter_avg_opinion", "weibo_weighted_opinion", "twitter_weighted_opinion", "weibo_user_avg_opinion", "twitter_user_avg_opinion"]`, saved to both `output_path` (parquet) and the same path with a `.csv` suffix. Task 11's `plot_figures.py` reads this exact column layout.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_prepare_data.py
from prepare_data import export


def test_export_merges_both_platforms_unsmoothed(tmp_path):
    weibo_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-10"],
        "avg_opinion": [0.5, 1.0],
        "weighted_opinion": [0.4, 0.9],
        "user_avg_opinion": [0.3, 0.8],
    })
    weibo_path = tmp_path / "weibo_daily_opinion.parquet"
    weibo_df.to_parquet(weibo_path, index=False)

    twitter_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-20"],
        "avg_opinion": [-0.5, -1.0],
        "weighted_opinion": [-0.4, -0.9],
        "user_avg_opinion": [-0.3, -0.8],
    })
    twitter_path = tmp_path / "twitter_daily_opinion.parquet"
    twitter_df.to_parquet(twitter_path, index=False)

    output_path = tmp_path / "figure_data.parquet"

    result = export(str(weibo_path), str(twitter_path), str(output_path))

    expected_columns = [
        "date",
        "weibo_avg_opinion",
        "twitter_avg_opinion",
        "weibo_weighted_opinion",
        "twitter_weighted_opinion",
        "weibo_user_avg_opinion",
        "twitter_user_avg_opinion",
    ]
    assert list(result.columns) == expected_columns
    assert set(result["date"]) == {"2024-03-01", "2024-03-10", "2024-03-20"}

    row = result[result["date"] == "2024-03-01"].iloc[0]
    assert row["weibo_avg_opinion"] == pytest.approx(0.5)
    assert row["twitter_avg_opinion"] == pytest.approx(-0.5)

    # 2024-03-10 has no twitter data -> NaN, not dropped, not zero
    row_weibo_only = result[result["date"] == "2024-03-10"].iloc[0]
    assert row_weibo_only["weibo_avg_opinion"] == pytest.approx(1.0)
    assert pd.isna(row_weibo_only["twitter_avg_opinion"])

    assert output_path.exists()
    assert output_path.with_suffix(".csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prepare_data.py -v`
Expected: FAIL — `export` not defined.

- [ ] **Step 3: Write the implementation**

```python
# append to prepare_data.py (before main())

def export(weibo_path: str, twitter_path: str, output_path: str) -> pd.DataFrame:
    weibo_df = pd.read_parquet(weibo_path).rename(
        columns={
            "avg_opinion": "weibo_avg_opinion",
            "weighted_opinion": "weibo_weighted_opinion",
            "user_avg_opinion": "weibo_user_avg_opinion",
        }
    )
    twitter_df = pd.read_parquet(twitter_path).rename(
        columns={
            "avg_opinion": "twitter_avg_opinion",
            "weighted_opinion": "twitter_weighted_opinion",
            "user_avg_opinion": "twitter_user_avg_opinion",
        }
    )

    merged = weibo_df.merge(twitter_df, on="date", how="outer")
    merged = merged[
        [
            "date",
            "weibo_avg_opinion",
            "twitter_avg_opinion",
            "weibo_weighted_opinion",
            "twitter_weighted_opinion",
            "weibo_user_avg_opinion",
            "twitter_user_avg_opinion",
        ]
    ]
    merged = merged.sort_values("date").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    merged.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Saved figure data to {output_path} ({len(merged)} dates)")
    return merged


def _export_cli():
    import config

    export(
        weibo_path=f"{config.OUTPUT_DIR}/weibo_daily_opinion.parquet",
        twitter_path=f"{config.OUTPUT_DIR}/twitter_daily_opinion.parquet",
        output_path=f"{config.OUTPUT_DIR}/figure_data.parquet",
    )
```

Update `main()` to register the new subcommand:

```python
def main():
    fire.Fire({"clean": _clean_cli, "export": _export_cli})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare_data.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add prepare_data.py tests/test_prepare_data.py
git commit -m "feat: add Stage 2 export command producing unsmoothed figure_data"
```

---

### Task 11: Stage 3 CLI — `plot_figures.py`

**Files:**
- Create: `plot_figures.py`
- Test: `tests/test_plot_figures.py`

**Interfaces:**
- Consumes: `figure_data` shaped like Task 10's `export()` output.
- Produces: `plot_figures.apply_sliding_window(df, metric, window_size=3) -> pd.Series`; `plot_figures.plot_metric(ax, figure_df, metric_name, ylabel, window_size=3, use_smoothing=True) -> pd.DataFrame` (columns `date, weibo, twitter`); `plot_figures.main(figure_data_path=None, output_dir=None, window_size=3, use_smoothing=True)` — writes one PDF + one companion CSV per metric (`avg_opinion`, `weighted_opinion`, `user_avg_opinion`) to `output_dir`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plot_figures.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from plot_figures import apply_sliding_window, main, plot_metric


def test_apply_sliding_window_centered_mean():
    df = pd.DataFrame({"date": pd.date_range("2024-03-01", periods=5), "metric": [1, 2, 3, 4, 5]})
    smoothed = apply_sliding_window(df, "metric", window_size=3)
    assert smoothed.tolist() == pytest.approx([1.5, 2.0, 3.0, 4.0, 4.5])


def test_plot_metric_without_smoothing_returns_raw_two_lines():
    figure_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-10"],
        "weibo_avg_opinion": [0.5, 1.0],
        "twitter_avg_opinion": [-0.5, -1.0],
    })
    fig, ax = plt.subplots()
    points = plot_metric(
        ax, figure_df, "avg_opinion", "Average Opinion", use_smoothing=False
    )
    plt.close(fig)

    assert list(points.columns) == ["date", "weibo", "twitter"]
    assert points["weibo"].tolist() == pytest.approx([0.5, 1.0])
    assert points["twitter"].tolist() == pytest.approx([-0.5, -1.0])


def test_main_writes_one_pdf_and_csv_per_metric(tmp_path):
    figure_df = pd.DataFrame({
        "date": pd.date_range("2024-03-01", periods=5).strftime("%Y-%m-%d"),
        "weibo_avg_opinion": [0.1, 0.2, 0.3, 0.4, 0.5],
        "twitter_avg_opinion": [-0.1, -0.2, -0.3, -0.4, -0.5],
        "weibo_weighted_opinion": [0.1, 0.2, 0.3, 0.4, 0.5],
        "twitter_weighted_opinion": [-0.1, -0.2, -0.3, -0.4, -0.5],
        "weibo_user_avg_opinion": [0.1, 0.2, 0.3, 0.4, 0.5],
        "twitter_user_avg_opinion": [-0.1, -0.2, -0.3, -0.4, -0.5],
    })
    figure_data_path = tmp_path / "figure_data.parquet"
    figure_df.to_parquet(figure_data_path, index=False)
    output_dir = tmp_path / "figures"

    main(figure_data_path=str(figure_data_path), output_dir=str(output_dir))

    for metric_name in ["avg_opinion", "weighted_opinion", "user_avg_opinion"]:
        pdf_path = output_dir / f"{metric_name}_comparison.pdf"
        csv_path = output_dir / f"{metric_name}_comparison.csv"
        assert pdf_path.exists()
        assert csv_path.exists()
        saved = pd.read_csv(csv_path)
        # exactly two lines: weibo vs twitter, no four-line variant
        assert list(saved.columns) == ["date", "weibo", "twitter"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plot_figures.py -v`
Expected: FAIL — `plot_figures.py` does not exist.

- [ ] **Step 3: Write the implementation**

```python
# plot_figures.py
"""Stage 3 CLI: plot the Weibo-vs-Twitter comparison figures.

figure_data on disk stays unsmoothed; sliding-window smoothing happens only
here, at plot time, and is never written back.

Usage:
    python plot_figures.py
    python plot_figures.py --window_size 5
    python plot_figures.py --use_smoothing False
"""

from pathlib import Path
from typing import Optional

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

WEIBO_COLOR = "#ff7333"
TWITTER_COLOR = "#20AEE6"

METRICS = [
    ("avg_opinion", "Average Opinion"),
    ("weighted_opinion", "LikeCount Weighted Opinion"),
    ("user_avg_opinion", "User-level Average Opinion"),
]


def apply_sliding_window(df: pd.DataFrame, metric: str, window_size: int = 3) -> pd.Series:
    df = df.sort_values("date")
    return df[metric].rolling(window=window_size, center=True, min_periods=1).mean()


def plot_metric(
    ax,
    figure_df: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    window_size: int = 3,
    use_smoothing: bool = True,
) -> pd.DataFrame:
    df = figure_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    weibo_col = f"weibo_{metric_name}"
    twitter_col = f"twitter_{metric_name}"

    if use_smoothing:
        weibo_values = apply_sliding_window(
            df.rename(columns={weibo_col: metric_name}), metric_name, window_size
        )
        twitter_values = apply_sliding_window(
            df.rename(columns={twitter_col: metric_name}), metric_name, window_size
        )
    else:
        weibo_values = df[weibo_col]
        twitter_values = df[twitter_col]

    ax.plot(df["date"], weibo_values, color=WEIBO_COLOR, linewidth=5, alpha=0.7, label="Weibo, China")
    ax.plot(df["date"], twitter_values, color=TWITTER_COLOR, linewidth=5, alpha=0.7, label="Twitter, USA")

    if metric_name == "weighted_opinion":
        ax.axhline(y=0, color="grey", linestyle="--", linewidth=2, zorder=0)

    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    return pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d").values,
            "weibo": weibo_values.values,
            "twitter": twitter_values.values,
        }
    )


def main(
    figure_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    window_size: int = 3,
    use_smoothing: bool = True,
):
    if figure_data_path is None or output_dir is None:
        import config

        figure_data_path = figure_data_path or f"{config.OUTPUT_DIR}/figure_data.parquet"
        output_dir = output_dir or f"{config.OUTPUT_DIR}/figures"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    figure_df = pd.read_parquet(figure_data_path)

    for metric_name, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        points = plot_metric(
            ax, figure_df, metric_name, ylabel, window_size=window_size, use_smoothing=use_smoothing
        )
        plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.15)

        output_path = Path(output_dir) / f"{metric_name}_comparison.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        points.to_csv(output_path.with_suffix(".csv"), index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_plot_figures.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add plot_figures.py tests/test_plot_figures.py
git commit -m "feat: add Stage 3 CLI plot_figures.py with plot-time-only smoothing"
```

---

### Task 12: End-to-end integration test across all three stages

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: `run_analysis.analyze` (Task 8), `prepare_data.clean` / `prepare_data.export` (Tasks 9, 10), `plot_figures.main` (Task 11). No new production code — this task only adds a test that wires the existing functions together to catch integration mismatches (e.g. column-name drift between stages) that per-module tests can't see.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_integration.py
import pandas as pd

from plot_figures import main as plot_main
from prepare_data import clean, export
from run_analysis import analyze


class ScriptedClient:
    """Deterministic opinion by post text, so the aggregate math is checkable."""

    def __init__(self, opinion_by_text):
        self._opinion_by_text = opinion_by_text

    def analyze_one(self, text):
        return {
            "opinion": self._opinion_by_text[text],
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
        }


def test_full_pipeline_weibo_and_twitter_to_plots(tmp_path):
    # --- Weibo input: two posts on 2024-03-01 ---
    weibo_dir = tmp_path / "weibo_input"
    weibo_dir.mkdir()
    pd.DataFrame({
        "weibo_id": ["w1", "w2"],
        "user_id": ["u1", "u2"],
        "weibo_content": ["weibo positive", "weibo negative"],
        "zan": [0, 0],
    }).to_parquet(weibo_dir / "2024-03-01.parquet", index=False)

    # --- Twitter input: two posts on 2024-03-01 ---
    twitter_dir = tmp_path / "twitter_input"
    twitter_dir.mkdir()
    pd.DataFrame({
        "id": ["t1", "t2"],
        "text": ["tweet positive", "tweet negative"],
        "likeCount": [0, 0],
        "author.id": ["a1", "a2"],
        "createdAt": [
            "Fri Mar 01 12:00:00 +0000 2024",
            "Fri Mar 01 13:00:00 +0000 2024",
        ],
    }).to_parquet(twitter_dir / "tweets_2024-03-01.parquet", index=False)

    output_dir = tmp_path / "output"

    # Stage 1
    weibo_results_path = output_dir / "analysis_results" / "weibo_opinion_results.csv"
    analyze(
        platform="weibo",
        input_dir=str(weibo_dir),
        filename_pattern="{date}.parquet",
        output_path=str(weibo_results_path),
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m",
        start_date="2024-03-01", end_date="2024-03-05", target_days=[1, 10, 20],
        client=ScriptedClient({"weibo positive": 2, "weibo negative": -2}),
    )

    twitter_results_path = output_dir / "analysis_results" / "twitter_opinion_results.csv"
    analyze(
        platform="twitter",
        input_dir=str(twitter_dir),
        filename_pattern="tweets_{date}.parquet",
        output_path=str(twitter_results_path),
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m",
        start_date="2024-03-01", end_date="2024-03-05", target_days=[1, 10, 20],
        client=ScriptedClient({"tweet positive": 1, "tweet negative": -1}),
    )

    # Stage 2
    weibo_daily_path = output_dir / "weibo_daily_opinion.parquet"
    clean(
        platform="weibo", input_dir=str(weibo_dir), filename_pattern="{date}.parquet",
        start_date="2024-03-01", end_date="2024-03-05", target_days=[1, 10, 20],
        opinion_results_path=str(weibo_results_path), output_path=str(weibo_daily_path),
    )

    twitter_daily_path = output_dir / "twitter_daily_opinion.parquet"
    clean(
        platform="twitter", input_dir=str(twitter_dir), filename_pattern="tweets_{date}.parquet",
        start_date="2024-03-01", end_date="2024-03-05", target_days=[1, 10, 20],
        opinion_results_path=str(twitter_results_path), output_path=str(twitter_daily_path),
    )

    figure_data_path = output_dir / "figure_data.parquet"
    figure_data = export(str(weibo_daily_path), str(twitter_daily_path), str(figure_data_path))

    row = figure_data[figure_data["date"] == "2024-03-01"].iloc[0]
    assert row["weibo_avg_opinion"] == pytest.approx(0.0)  # mean(2, -2)
    assert row["twitter_avg_opinion"] == pytest.approx(0.0)  # mean(1, -1)

    # Stage 3
    figures_dir = output_dir / "figures"
    plot_main(figure_data_path=str(figure_data_path), output_dir=str(figures_dir))

    for metric_name in ["avg_opinion", "weighted_opinion", "user_avg_opinion"]:
        assert (figures_dir / f"{metric_name}_comparison.pdf").exists()
        assert (figures_dir / f"{metric_name}_comparison.csv").exists()
```

Add `import pytest` at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration.py -v`
Expected: at this point in the plan all underlying pieces already exist (Tasks 1–11 done), so this should actually PASS immediately if everything is wired correctly — treat any FAIL here as a real integration bug (e.g. a column name mismatch between `clean()`'s output and `export()`'s expected input) and fix it before moving on, rather than assuming the test itself is wrong.

- [ ] **Step 3: Fix any integration mismatch found**

If Step 2 fails, the fix is in whichever of `run_analysis.py` / `prepare_data.py` /
`plot_figures.py` has the mismatched column name or signature — not in the test.
Re-run `pytest tests/test_integration.py -v` after each fix until it passes.

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: all tests across all modules PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test across all three pipeline stages"
```
