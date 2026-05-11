from __future__ import annotations

import json
from pathlib import Path


SAMPLE_QUESTIONS = [
    {
        "question": "Explain the difference between although and even though with one example.",
        "expected_answer": "Both introduce contrast. Even though often sounds stronger or more conversational.",
        "expected_context_keywords": ["contrast", "although", "even though"],
        "category": "grammar",
        "difficulty": "easy",
    },
    {
        "question": "Correct this sentence: I am agree with you.",
        "expected_answer": "I agree with you.",
        "expected_context_keywords": ["I agree with you"],
        "category": "correction",
        "difficulty": "easy",
    },
    {
        "question": "Give me a short interview answer about my experience with MLOps.",
        "expected_answer": "A concise first-person answer mentioning production ML, monitoring, and deployment.",
        "expected_context_keywords": ["MLOps", "monitoring", "deployment"],
        "category": "interview",
        "difficulty": "medium",
    },
]


def write_sample_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in SAMPLE_QUESTIONS:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    path = write_sample_dataset(Path("data/evaluation/datasets/sample_questions.jsonl"))
    print(f"Wrote sample dataset: {path}")


if __name__ == "__main__":
    main()
