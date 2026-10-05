#!/usr/bin/env python3
"""Zero-shot classify GitHub issues using DeBERTa-v3-base-mnli (ONNX).

Reads ``scripts/issue_triage/issuesjsonl``, assigns the top-2 most likely
categories per issue (or fewer, if only one exceeds the minimum score), and
writes the result back in-place (adding a ``category`` key).

Usage:
    python scripts/issue_triage/classify_1.py            # skip already-classified
    python scripts/issue_triage/classify_1.py --overwrite  # reclassify everything
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

HERE = Path(__file__).parent
JSONL = HERE / "issues.jsonl"
MODEL_ID = "Xenova/DeBERTa-v3-base-mnli"
ONNX_FILE = "onnx/model_fp16.onnx"
TOKENIZER_ID = "Xenova/DeBERTa-v3-base-mnli"
MAX_LENGTH = 512
TOP_K = 2  # assign up to this many categories
MIN_SCORE = 0.15  # ignore categories below this entailment score (soft floor)
ISSUES_PER_BATCH = 4

CATEGORIES = [
    "model support",
    "LoRA and training",
    "model download and loading",
    "bug or error",
    "feature request",
    "CLI, API, or UX",
    "performance, speed, or memory",
    "quantization",
    "installation and setup",
    "documentation",
    "CI and tests",
]

log = logging.getLogger(__name__)


def build_hypotheses(categories: list[str]) -> list[str]:
    """Standard zero-shot template."""
    return [f"This text is about {c}." for c in categories]


def load_model() -> tuple[ort.InferenceSession, PreTrainedTokenizerBase]:
    """Download ONNX model (cached) and load tokenizer."""
    log.info("Loading tokenizer %s …", TOKENIZER_ID)
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(TOKENIZER_ID))

    log.info("Loading ONNX model %s/%s …", MODEL_ID, ONNX_FILE)
    model_path = hf_hub_download(repo_id=MODEL_ID, filename=ONNX_FILE)
    providers = ["CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CoreMLExecutionProvider")
    log.info("Using ONNX Runtime providers: %s", providers)
    session = ort.InferenceSession(model_path, providers=providers)
    return session, tokenizer


def classify_batch(
    texts: list[str],
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    hypotheses: list[str],
    categories: list[str],
) -> list[list[str]]:
    """Return top-K categories by entailment-vs-contradiction score for each text."""
    premises = [text[: MAX_LENGTH * 4] for text in texts]
    inputs = tokenizer(
        [premise for premise in premises for _ in hypotheses],
        hypotheses * len(premises),
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # MNLI logits: [contradiction, neutral, entailment]  (index 0, 1, 2)
    logits = np.asarray(
        session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})[0]
    )

    # 2-way score: entailment / (entailment + contradiction), ignoring neutral
    logits = logits.reshape(len(texts), len(categories), -1)
    cont = logits[:, :, 0]
    ent = logits[:, :, 2]
    scores = softmax(np.stack([cont, ent], axis=1), axis=1)[:, 1]

    assignments = []
    for issue_scores in scores:
        ranked = sorted(zip(issue_scores, categories), reverse=True)
        assigned = [label for score, label in ranked[:TOP_K] if float(score) >= MIN_SCORE]
        assignments.append(assigned if assigned else ["other"])

    return assignments


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def read_issues(path: Path) -> list[dict]:
    issues = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                issues.append(json.loads(line))
    return issues


def write_issues(path: Path, issues: list[dict]) -> None:
    with open(path, "w") as f:
        for issue in issues:
            f.write(json.dumps(issue, ensure_ascii=False) + "\n")


def category_totals(issues: list[dict]) -> list[tuple[str, int]]:
    totals: Counter[str] = Counter()
    for issue in issues:
        totals.update(issue.get("category", []))
    return sorted(totals.items())


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify mflux GitHub issues")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="Reclassify issues that already have a category")
    mode.add_argument("--totals", action="store_true", help="Show category totals without classifying issues")
    parser.add_argument("--json", action="store_true", help="Output totals as JSON")
    args = parser.parse_args()
    if args.json and not args.totals:
        parser.error("--json requires --totals")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    issues = read_issues(JSONL)
    if args.totals:
        totals = category_totals(issues)
        if args.json:
            print(json.dumps([{"category": category, "count": count} for category, count in totals]))
        else:
            for category, count in totals:
                print(f"{category}: {count}")
        return 0

    total = len(issues)

    to_classify = [i for i in issues if args.overwrite or "category" not in i]
    already = total - len(to_classify)
    log.info("%d issues total, %d already classified, %d to classify", total, already, len(to_classify))

    if not to_classify:
        log.info("Nothing to do.")
        return 0

    session, tokenizer = load_model()
    hypotheses = build_hypotheses(CATEGORIES)

    for start in range(0, len(to_classify), ISSUES_PER_BATCH):
        issue_batch = to_classify[start : start + ISSUES_PER_BATCH]
        texts = [f"{issue['title']}\n{issue['body']}" for issue in issue_batch]
        labels_by_issue = classify_batch(texts, session, tokenizer, hypotheses, CATEGORIES)
        for issue, labels in zip(issue_batch, labels_by_issue, strict=True):
            issue["category"] = labels
        if start + len(issue_batch) < len(to_classify):
            log.info("  %d/%d", start + len(issue_batch), len(to_classify))

    write_issues(JSONL, issues)
    log.info("Wrote %s", JSONL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
