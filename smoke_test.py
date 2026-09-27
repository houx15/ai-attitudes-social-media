"""End-to-end smoke test of all three pipeline stages on generated mock data.

Everything goes under smoke_output/ (wiped and recreated on each run). The real
CLI commands run with a throwaway config.py placed there, so your own config.py
is never modified.

    python smoke_test.py          # offline: Stage 1 talks to a local fake OpenRouter server
    python smoke_test.py --live   # Stage 1 calls the real OpenRouter API with the key/model
                                  # from your config.py (~100 short requests)
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from openrouter_client import normalize_opinion
from prompts import SYSTEM_PROMPT

REPO = Path(__file__).resolve().parent
SMOKE_DIR = REPO / "smoke_output"
DATA_DIR = SMOKE_DIR / "data"
OUTPUT_DIR = SMOKE_DIR / "output"

TARGET_DATES = ["2024-03-01", "2024-03-10", "2024-03-20", "2024-04-01"]
OFF_DAY = "2024-03-05"  # not a target day, so its file must never be read
# Same layout as the real data: Weibo's first sampled day was crawled a day early.
DATE_SUBSTITUTIONS = {"weibo": {"2024-03-01": "2024-02-29"}, "twitter": {}}
# Twitter metrics are always restricted to these users; tu6 stands in for non-US authors.
US_TWITTER_USERS = ["tu1", "tu2", "tu3", "tu4", "tu5"]
POSTS_PER_DAY = 12
CANNOT_TELL = "cannot tell"
FLAKY = "[flaky] "  # fake server answers 500 the first time it sees this post
BADJSON = "[badjson] "  # fake server answers prose instead of JSON, every time
MOCK_MODEL = "mock/deepseek-v4.1-flash"
METRICS = ["avg_opinion", "weighted_opinion", "user_avg_opinion"]

TEMPLATES = {
    "weibo": {
        2: [
            "AI大模型太棒了，写代码效率提高了好几倍，未来可期！",
            "用DeepSeek做医学文献综述，真是颠覆性的工具，强烈推荐",
            "人工智能会让每个人的生活都更便利，我完全支持",
        ],
        1: [
            "试了一下豆包，挺有用的，虽然偶尔会出错",
            "ChatGPT帮我改简历还不错，总体有帮助",
            "AI写作工具还行，能省点时间",
        ],
        0: [
            "AI有利有弊吧，看怎么用",
            "大模型到底会带来什么，现在还说不清",
            "关于人工智能的讨论越来越多了，各有各的道理",
        ],
        -1: [
            "有点担心AI生成的假新闻越来越多",
            "孩子用AI写作业，我有点不放心",
            "大模型老是一本正经地胡说八道，不太敢信",
        ],
        -2: [
            "AI迟早会取代大量工作岗位，失业潮要来了，太可怕了",
            "人工智能泄露隐私的风险太大了，坚决反对",
            "AI换脸诈骗太猖獗，这技术弊大于利",
        ],
        CANNOT_TELL: [
            "今天天气真好，出去散步了",
            "晚饭吃了火锅，好开心",
            "周末去看了场电影",
        ],
    },
    "twitter": {
        2: [
            "AI coding assistants are an absolute game changer, my productivity doubled!",
            "Using AI for protein folding research is incredible, the future of medicine is here",
            "LLMs make learning so much easier, huge win for everyone",
        ],
        1: [
            "Claude is pretty useful for drafting emails, though it makes mistakes sometimes",
            "ChatGPT helped me debug today, generally helpful",
            "AI summaries are decent, they save me some time",
        ],
        0: [
            "AI has pros and cons, depends how you use it",
            "Not sure yet what LLMs will mean for society",
            "Interesting debate about AI regulation today, both sides have points",
        ],
        -1: [
            "A bit worried about AI-generated misinformation flooding my feed",
            "Not comfortable with kids using AI for homework",
            "Hallucinations make me hesitant to trust these chatbots",
        ],
        -2: [
            "AI is going to wipe out millions of jobs, this is dangerous",
            "AI surveillance is a massive privacy threat, ban it",
            "Deepfake scams are out of control, this tech does more harm than good",
        ],
        CANNOT_TELL: [
            "Beautiful weather today, went for a walk",
            "Great tacos for lunch",
            "Watched a movie this weekend",
        ],
    },
}
TEXT_TO_OPINION = {
    text: category
    for platform_templates in TEMPLATES.values()
    for category, texts in platform_templates.items()
    for text in texts
}


def category_weights(platform, day_index):
    # Weibo drifts positive and Twitter drifts negative over time, so the two
    # lines in the figures are visibly different.
    weights = {2: 1, 1: 2, 0: 2, -1: 2, -2: 1, CANNOT_TELL: 1}
    if platform == "weibo":
        weights[2] += day_index
        weights[1] += day_index
    else:
        weights[-2] += day_index
        weights[-1] += day_index
    return weights


def generate_mock_data(fault_injection, seed=7):
    """Write mock per-day parquet files; return the ground truth, one row per post."""
    rng = random.Random(seed)
    truth = []
    for platform in ("weibo", "twitter"):
        (DATA_DIR / platform).mkdir(parents=True, exist_ok=True)
        platform_dates = [DATE_SUBSTITUTIONS[platform].get(d, d) for d in TARGET_DATES]
        for day_index, date in enumerate(platform_dates + [OFF_DAY]):
            is_target = date != OFF_DAY
            rows = []
            for n in range(POSTS_PER_DAY):
                weights = category_weights(platform, day_index)
                category = rng.choices(list(weights), weights=list(weights.values()))[0]
                text = rng.choice(TEMPLATES[platform][category])
                expected = np.nan if category == CANNOT_TELL else float(category)
                if fault_injection and is_target and day_index == 0 and n == 0:
                    text, expected = BADJSON + text, np.nan
                if fault_injection and is_target and day_index == 0 and n == 1:
                    text = FLAKY + text
                rows.append(
                    {
                        "id": f"{platform[0]}{date.replace('-', '')}{n:03d}",
                        "text": text,
                        "user": f"{platform[0]}u{rng.randint(1, 6)}",
                        "likes": rng.choice([0, 0, 1, 3, 8, 20, 150]),
                        "time": datetime.strptime(date, "%Y-%m-%d")
                        + timedelta(seconds=rng.randint(0, 86399)),
                        "category": category,
                        "expected_opinion": expected,
                        "sent": is_target,
                    }
                )
            if day_index == 0:
                # Upstream shards can repeat a post; the pipeline must count it once.
                rows.append({**rows[2], "sent": False})
                # Legacy behaviour: Weibo sends empty/missing text as "", Twitter
                # skips it.
                for k, empty in enumerate(["", None]):
                    rows.append(
                        {**rows[3], "id": f"{platform[0]}empty{k}", "text": empty,
                         "expected_opinion": np.nan, "sent": platform == "weibo"}
                    )
                # Last second of the day in UTC still belongs to that date.
                rows[4]["time"] = datetime.strptime(date, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59
                )
            _write_day_file(platform, date, rows)
            truth.extend({**row, "platform": platform, "date": date} for row in rows)
    return pd.DataFrame(truth)


def _write_day_file(platform, date, rows):
    if platform == "weibo":
        df = pd.DataFrame(
            {
                "weibo_id": [r["id"] for r in rows],
                "user_id": [int(r["user"][2:]) + 1000 for r in rows],
                "is_retweet": ["0"] * len(rows),
                "weibo_content": [r["text"] for r in rows],
                "zhuan": [0] * len(rows),
                "ping": [0] * len(rows),
                "zan": [r["likes"] for r in rows],
                "time_stamp": [r["time"].timestamp() for r in rows],
            }
        )
        df.to_parquet(DATA_DIR / "weibo" / f"{date}.parquet", index=False)
    else:
        df = pd.DataFrame(
            {
                "id": [r["id"] for r in rows],
                "text": [r["text"] for r in rows],
                "likeCount": [r["likes"] for r in rows],
                "retweetCount": [0] * len(rows),
                "author.id": [r["user"] for r in rows],
                "createdAt": [r["time"].strftime("%a %b %d %H:%M:%S +0000 %Y") for r in rows],
                "lang": ["en"] * len(rows),
            }
        )
        df.to_parquet(DATA_DIR / "twitter" / f"tweets_{date}.parquet", index=False)


class FakeOpenRouter:
    """Minimal stand-in for OpenRouter's /chat/completions endpoint."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []
        self.flaky_failed = set()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def respond(self, model, system_prompt, text, reasoning):
        with self.lock:
            self.requests.append(
                {"model": model, "system": system_prompt, "text": text, "reasoning": reasoning}
            )
            if text.startswith(FLAKY) and text not in self.flaky_failed:
                self.flaky_failed.add(text)
                return 500, None
        if text.startswith(BADJSON):
            return 200, "I think this post is mostly positive about AI."
        category = TEXT_TO_OPINION.get(text.replace(FLAKY, ""), CANNOT_TELL)
        return 200, json.dumps({"opinion": category})

    def _make_handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                messages = {m["role"]: m["content"] for m in body["messages"]}
                text = messages["user"].removeprefix("Post text: ")
                status, content = fake.respond(
                    body["model"], messages["system"], text, body.get("reasoning")
                )
                if status != 200:
                    self._send(status, {"error": {"message": "mock transient failure"}})
                    return
                self._send(
                    200,
                    {
                        "id": "mock",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 400,
                            "completion_tokens": 6,
                            "total_tokens": 406,
                            "prompt_tokens_details": {"cached_tokens": 384},
                        },
                    },
                )

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


def write_smoke_config(live, base_url=None):
    if live:
        # Re-export the real key/model from config.py rather than copying the
        # secret into another file.
        credentials = (
            "import importlib.util as _util\n"
            f"_spec = _util.spec_from_file_location('_real_config', {str(REPO / 'config.py')!r})\n"
            "_real = _util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_real)\n"
            "OPENROUTER_API_KEY = _real.OPENROUTER_API_KEY\n"
            "OPENROUTER_BASE_URL = _real.OPENROUTER_BASE_URL\n"
            "OPENROUTER_MODEL = _real.OPENROUTER_MODEL\n"
            "BACKOFF_BASE_SECONDS = 2.0\n"
        )
    else:
        credentials = (
            "OPENROUTER_API_KEY = 'mock-key'\n"
            f"OPENROUTER_BASE_URL = {base_url!r}\n"
            f"OPENROUTER_MODEL = {MOCK_MODEL!r}\n"
            "BACKOFF_BASE_SECONDS = 0.1\n"
        )
    (SMOKE_DIR / "config.py").write_text(
        '"""Throwaway config written by smoke_test.py."""\n'
        + credentials
        + f"START_DATE = {TARGET_DATES[0]!r}\n"
        f"END_DATE = {TARGET_DATES[-1]!r}\n"
        "TARGET_DAYS = [1, 10, 20]\n"
        f"DATE_SUBSTITUTIONS = {DATE_SUBSTITUTIONS!r}\n"
        f"WEIBO_INPUT_DIR = {str(DATA_DIR / 'weibo')!r}\n"
        "WEIBO_FILENAME_PATTERN = '{date}.parquet'\n"
        f"TWITTER_INPUT_DIR = {str(DATA_DIR / 'twitter')!r}\n"
        "TWITTER_FILENAME_PATTERN = 'tweets_{date}.parquet'\n"
        f"OUTPUT_DIR = {str(OUTPUT_DIR)!r}\n"
        f"TWITTER_US_USERIDS_PATH = {str(DATA_DIR / 'us_userids.json')!r}\n"
        "MAX_WORKERS = 4\n"
        "MAX_RETRIES = 3\n"
        "REQUEST_TIMEOUT = 60\n"
    )


def run_cli(*args):
    print(f"\n=== python -m {' '.join(args)} ===", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    # With -m, the working directory (smoke_output/) comes first on sys.path, so
    # `import config` picks up the throwaway config instead of the real one.
    subprocess.run([sys.executable, "-m", *args], cwd=SMOKE_DIR, env=env, check=True)


def read_results(platform):
    return pd.read_csv(
        OUTPUT_DIR / "analysis_results" / f"{platform}_opinion_results.csv",
        dtype=str,
        keep_default_na=False,
    )


def expected_daily_metrics(truth, platform):
    posts = truth[(truth["platform"] == platform) & truth["sent"]]
    if platform == "twitter":
        posts = posts[posts["user"].isin(US_TWITTER_USERS)]
    posts = posts.dropna(subset=["expected_opinion"]).assign(
        weight=lambda d: d["likes"] + 1,
        weighted=lambda d: d["expected_opinion"] * (d["likes"] + 1),
    )
    by_date = posts.groupby("date")
    return pd.DataFrame(
        {
            "avg_opinion": by_date["expected_opinion"].mean(),
            "weighted_opinion": by_date["weighted"].sum() / by_date["weight"].sum(),
            "user_avg_opinion": posts.groupby(["date", "user"])["expected_opinion"]
            .mean()
            .groupby("date")
            .mean(),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--live", action="store_true", help="call the real OpenRouter API")
    live = parser.parse_args().live

    if live:
        if not (REPO / "config.py").exists():
            sys.exit("--live needs config.py (cp config.example.py config.py, then add your key)")
        sys.path.insert(0, str(REPO))
        import config

        if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY.startswith("YOUR_"):
            sys.exit("--live: set OPENROUTER_API_KEY in config.py first")

    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    SMOKE_DIR.mkdir()
    truth = generate_mock_data(fault_injection=not live)
    (DATA_DIR / "us_userids.json").write_text(json.dumps(US_TWITTER_USERS))
    fake = None if live else FakeOpenRouter()
    write_smoke_config(live, base_url=None if live else fake.base_url)

    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # Stage 1, then a second pass to exercise resume.
    for platform in ("weibo", "twitter"):
        run_cli("run_analysis", platform)
    first = {p: read_results(p) for p in ("weibo", "twitter")}
    requests_after_first = len(fake.requests) if fake else None
    for platform in ("weibo", "twitter"):
        run_cli("run_analysis", platform)
    second = {p: read_results(p) for p in ("weibo", "twitter")}

    for platform in ("weibo", "twitter"):
        expected_ids = set(truth[(truth["platform"] == platform) & truth["sent"]]["id"])
        got = first[platform]
        check(
            f"[{platform}] each target-day post sent exactly once "
            "(off-day file and duplicate skipped; empty text sent for Weibo only, as before)",
            set(got["id"]) == expected_ids and len(got) == len(expected_ids),
            f"{len(got)} result rows, {len(expected_ids)} expected posts",
        )
        failed = got[got["opinion"].map(normalize_opinion).isna()]
        appended = second[platform].iloc[len(got):]
        check(
            f"[{platform}] rerunning Stage 1 retries only the failed rows",
            set(appended["id"]) == set(failed["id"]) and len(appended) == len(failed),
            f"{len(failed)} failed after first pass, {len(appended)} re-sent",
        )

    if fake:
        check(
            "every request sent the canonical SYSTEM_PROMPT, the configured model, and reasoning off",
            all(r["system"] == SYSTEM_PROMPT and r["model"] == MOCK_MODEL
                and r["reasoning"] == {"effort": "none"} for r in fake.requests),
            f"{len(fake.requests)} requests",
        )
        check(
            "a transient HTTP 500 was retried and the post still got a label",
            all(
                normalize_opinion(v) is not None
                for p in ("weibo", "twitter")
                for v in first[p].loc[
                    first[p]["id"].isin(truth[truth["text"].fillna("").str.startswith(FLAKY)]["id"]),
                    "opinion",
                ]
            ),
        )
        check(
            "token usage (incl. cached tokens nested in prompt_tokens_details) is recorded",
            all((first[p]["cached_tokens"].astype(int) > 0).any() for p in ("weibo", "twitter")),
        )
        check(
            "resume sent only the non-JSON posts again",
            all(r["text"].startswith(BADJSON) for r in fake.requests[requests_after_first:]),
            f"{len(fake.requests) - requests_after_first} requests on the second pass",
        )

    # Stages 2 and 3.
    for platform in ("weibo", "twitter"):
        run_cli("prepare_data", "clean", "--platform", platform)
    run_cli("prepare_data", "export")
    run_cli("plot_figures")

    figure_data = pd.read_parquet(OUTPUT_DIR / "figure_data.parquet")
    expected_dates = sorted(set(truth.loc[truth["sent"], "date"]))
    check(
        "figure_data has one row per sampled date (incl. Weibo's substituted 2024-02-29) "
        "and the 7 expected columns",
        list(figure_data["date"]) == expected_dates and figure_data.shape[1] == 7,
        f"dates {list(figure_data['date'])}",
    )
    figures = OUTPUT_DIR / "figures"
    points_files = list(figures.glob("user_avg_opinion_comparison_smoothed3d_*.csv"))
    points = pd.read_csv(points_files[0]).set_index("date")
    check(
        "smoothing invents no value where a platform has no data "
        "(Twitter on 2024-02-29, Weibo on 2024-03-01)",
        pd.isna(points.loc["2024-02-29", "twitter"]) and pd.isna(points.loc["2024-03-01", "weibo"])
        and points["weibo"].notna().sum() == 4 and points["twitter"].notna().sum() == 4,
    )
    if not live:
        mismatches = []
        for platform in ("weibo", "twitter"):
            expected = expected_daily_metrics(truth, platform)
            for metric in METRICS:
                got = figure_data.set_index("date")[f"{platform}_{metric}"]
                if not np.allclose(got.loc[expected.index], expected[metric]):
                    mismatches.append(f"{platform}_{metric}")
        check(
            "all 6 daily series in figure_data match independently computed ground truth "
            "(so export is unsmoothed, the math is right, and Twitter keeps only US users)",
            not mismatches,
            ", ".join(mismatches),
        )
    check(
        "Stage 3 wrote a PDF and a points CSV for each of the 3 metrics (legacy file naming)",
        all(len(list(figures.glob(f"{m}_comparison_smoothed3d_*{ext}"))) == 1
            for m in METRICS for ext in (".pdf", ".csv")),
    )

    if live:
        print("\n=== Live model labels vs. planted intent ===")
        for platform in ("weibo", "twitter"):
            results = second[platform].drop_duplicates("id", keep="last")
            labels = results.set_index("id")["opinion"].map(normalize_opinion)
            posts = truth[(truth["platform"] == platform) & truth["sent"]].set_index("id")
            valid = labels.dropna()
            planted = posts.loc[valid.index, "category"]
            exact = (valid.astype(str) == planted.astype(str)).mean()
            numeric = [(float(v), float(c)) for v, c in zip(valid, planted)
                       if v != CANNOT_TELL and c != CANNOT_TELL]
            same_sign = np.mean([np.sign(v) == np.sign(c) for v, c in numeric]) if numeric else float("nan")
            tokens = results[["prompt_tokens", "completion_tokens", "cached_tokens"]].astype(int).sum()
            avg_completion = tokens["completion_tokens"] / max(len(results), 1)
            print(
                f"[{platform}] {len(valid)}/{len(posts)} labeled, exact match {exact:.0%}, "
                f"same direction {same_sign:.0%}, tokens {dict(tokens)}"
            )
            check(
                f"[{platform}] reasoning is off (few output tokens per post)",
                avg_completion < 50,
                f"{avg_completion:.0f} output tokens per post on average",
            )
            check(f"[{platform}] the live model returned usable labels", len(valid) > 0,
                  "check OPENROUTER_MODEL slug and key" if len(valid) == 0 else "")

    print("\n=== Smoke test results ===")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    print(f"\nOutputs in {OUTPUT_DIR}")
    if fake:
        fake.server.shutdown()
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


if __name__ == "__main__":
    main()
