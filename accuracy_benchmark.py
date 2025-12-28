import argparse
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import httpx


TESTS: List[Dict[str, Any]] = [
    {"name": "math_1", "question": "Compute 19*23 and respond with just the number.", "expect": r"\b437\b"},
    {"name": "math_2", "question": "What is 7*8?", "expect": r"\b56\b"},
    {"name": "math_3", "question": "Compute (12+15)*3.", "expect": r"\b81\b"},
    {"name": "math_4", "question": "What is 144/12?", "expect": r"\b12\b"},
    {"name": "math_5", "question": "Compute 125*37.", "expect": r"\b4625\b"},
    {"name": "math_6", "question": "What is 2 to the 8th power?", "expect": r"\b256\b"},
    {"name": "math_7", "question": "Solve for x: 3x + 9 = 0.", "expect": r"-\\s*3\\b"},
    {
        "name": "math_8",
        "question": "Simplify 20/50. Answer as a fraction or decimal.",
        "expect": lambda a: re.search(r"\\b2/5\\b", a) or re.search(r"\\b0\\.4\\b", a),
    },
    {"name": "logic_1", "question": "If A is before B and B is before C, who is last?", "expect": r"\bC\b"},
    {"name": "logic_2", "question": "Reverse the string: 'stressed'.", "expect": r"\bdesserts\b"},
    {"name": "logic_3", "question": "If today is Monday, what day is two days later?", "expect": r"\bWednesday\b"},
    {"name": "logic_4", "question": "Which is larger: 0.3 or 0.27?", "expect": r"\b0\\.3(0)?\b"},
    {"name": "logic_5", "question": "Is 29 a prime number? Answer yes or no.", "expect": r"\byes\b"},
    {"name": "fact_1", "question": "What is the capital of France?", "expect": r"\bparis\b"},
    {"name": "fact_2", "question": "What planet is known as the Red Planet?", "expect": r"\bmars\b"},
    {"name": "fact_3", "question": "What is the chemical symbol for water?", "expect": r"\bh2o\b"},
    {"name": "fact_4", "question": "How many minutes are in 2 hours?", "expect": r"\b120\b"},
    {"name": "convert_1", "question": "Convert 5 kilometers to meters.", "expect": r"\b5000\b"},
    {"name": "convert_2", "question": "Convert 3.5 hours to minutes.", "expect": r"\b210\b"},
    {"name": "convert_3", "question": "Convert 2.5 kilograms to grams.", "expect": r"\b2500\b"},
    {"name": "unit_1", "question": "Boiling point of water at sea level in Celsius?", "expect": r"\b100\b"},
    {"name": "unit_2", "question": "How many seconds are in 3 minutes?", "expect": r"\b180\b"},
    {"name": "string_1", "question": "Make this lowercase: 'HELLO WORLD'.", "expect": r"\bhello world\b"},
    {"name": "string_2", "question": "Reverse the string: 'drawer'.", "expect": r"\breward\b"},
    {"name": "string_3", "question": "Count the letters in 'bookkeeper'.", "expect": r"\b10\b"},
    {"name": "date_1", "question": "How many days are in a non-leap year?", "expect": r"\b365\b"},
    {"name": "date_2", "question": "How many hours are in 2.5 days?", "expect": r"\b60\b"},
    {"name": "percent_1", "question": "What is 20% of 250?", "expect": r"\b50\b"},
    {"name": "percent_2", "question": "What is 15% of 80?", "expect": r"\b12\b"},
    {"name": "average_1", "question": "What is the average of 4, 9, and 17?", "expect": r"\b10\b"},
    {"name": "algebra_1", "question": "If y = 2x and x = 7, what is y?", "expect": r"\b14\b"},
    {"name": "code_1", "question": "What is the output of: print(2**5)?", "expect": r"\b32\b"},
]


def evaluate_answer(answer: str, expect: Any) -> bool:
    if callable(expect):
        return bool(expect(answer))
    if isinstance(expect, str):
        return re.search(expect, answer, re.IGNORECASE) is not None
    return False


def resolve_base_url(base_url: Optional[str], port: Optional[int]) -> str:
    if base_url:
        return base_url.rstrip("/")
    port_value = port or 8000
    return f"http://127.0.0.1:{port_value}"


def run_test(
    client: httpx.Client,
    base_url: str,
    question: str,
    model_tier: str,
    reasoning_mode: str,
    manual_level: str,
    strict_mode: bool,
    timeout_s: int,
) -> str:
    payload: Dict[str, Any] = {
        "question": question,
        "model_tier": model_tier,
        "reasoning_mode": reasoning_mode,
        "strict_mode": strict_mode,
    }
    if reasoning_mode == "manual":
        payload["manual_level"] = manual_level
    resp = client.post(f"{base_url}/api/run", json=payload)
    resp.raise_for_status()
    run_id = resp.json().get("run_id")
    if not run_id:
        raise RuntimeError("Run did not return run_id")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run_resp = client.get(f"{base_url}/api/run/{run_id}")
        run_resp.raise_for_status()
        run = run_resp.json()
        status = run.get("status") or ""
        answer = run.get("final_answer") or ""
        if status and status.lower() not in ("running", "in_progress") and answer:
            return str(answer)
        if status and status.lower() in ("stopped", "error") and answer:
            return str(answer)
        time.sleep(1.5)
    raise TimeoutError("Run did not finish before timeout")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a basic accuracy benchmark against LocalPro.")
    parser.add_argument("--base-url", help="Base URL (ex: http://127.0.0.1:8000).")
    parser.add_argument("--port", type=int, help="Port override if base URL is not set.")
    parser.add_argument("--model-tier", default="pro", choices=["fast", "pro", "deep", "auto"])
    parser.add_argument("--reasoning-mode", default="auto", choices=["auto", "manual"])
    parser.add_argument("--manual-level", default="MED", choices=["LOW", "MED", "HIGH", "ULTRA"])
    parser.add_argument("--strict", action="store_true", help="Enable strict verification mode.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-test timeout in seconds.")
    args = parser.parse_args()

    base_url = resolve_base_url(args.base_url, args.port)
    timeout = httpx.Timeout(10.0, read=None)

    total = 0
    passed = 0
    failures: List[str] = []

    with httpx.Client(timeout=timeout) as client:
        for test in TESTS:
            total += 1
            name = test["name"]
            question = test["question"]
            expect = test["expect"]
            try:
                answer = run_test(
                    client,
                    base_url,
                    question,
                    args.model_tier,
                    args.reasoning_mode,
                    args.manual_level,
                    args.strict,
                    args.timeout,
                )
            except Exception as exc:
                failures.append(f"{name}: error ({exc})")
                continue
            if evaluate_answer(answer, expect):
                passed += 1
                print(f"{name}: PASS")
            else:
                failures.append(f"{name}: FAIL (answer={answer!r})")
                print(f"{name}: FAIL")

    accuracy = (passed / total * 100.0) if total else 0.0
    print(f"Accuracy: {passed}/{total} ({accuracy:.1f}%)")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
