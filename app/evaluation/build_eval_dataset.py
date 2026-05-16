from __future__ import annotations

import json
from pathlib import Path


SAMPLE_QUESTIONS = [
    {
        "id": "grammar_001",
        "input": "I have a difficult question about programming with AI.",
        "expected_output": "I have a difficult question about programming with AI.",
        "task_type": "grammar_correction",
        "tags": ["english", "grammar"],
        "metadata": {"difficulty": "easy"},
    },
    {
        "id": "grammar_002",
        "input": "Correct this sentence: I am agree with you.",
        "expected_output": "I agree with you.",
        "task_type": "grammar_correction",
        "tags": ["english", "grammar"],
        "metadata": {"difficulty": "easy"},
    },
    {
        "id": "conversation_001",
        "input": "Give me a short interview answer about my experience with MLOps.",
        "expected_output": "I have worked on production ML systems, including deployment, monitoring, and continuous improvement.",
        "task_type": "conversation",
        "tags": ["english", "interview"],
        "reference_context": "The answer should mention production ML systems, deployment, monitoring, and improvement.",
        "metadata": {"difficulty": "medium"},
    },
]


def write_sample_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in SAMPLE_QUESTIONS:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    paths = [
        write_sample_dataset(Path("evals/english_practice_eval.jsonl")),
        write_sample_dataset(Path("data/evaluation/datasets/sample_questions.jsonl")),
    ]
    for path in paths:
        print(f"Wrote sample dataset: {path}")


if __name__ == "__main__":
    main()
