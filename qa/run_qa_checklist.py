"""Run the hybrid QA checklist against Butler and score every case.

Grading model (decided with the user):
  - The local LLM is the AUTHORITY on semantic content (judge-primary).
  - Deterministic STRICT GATES independently fail a case for the unambiguous
    things only: HTTP status, AUTO routing modes, and exact refusal/fallback
    strings. This keeps routing/refusal discipline strict without letting a
    brittle substring miss fail an otherwise-correct answer.
  - `must_contain` / `must_not_contain` are computed and handed to the judge as
    hints. With --skip-judge they become the deterministic content grader.

Execution is in-process via Flask's test_client, so app.py does NOT need to be
running. Neo4j must be up for graph cases (otherwise those cases are ERROR).

Usage:
  python qa/run_qa_checklist.py                 # all 32 cases, judge on
  python qa/run_qa_checklist.py --mode sql      # one mode only
  python qa/run_qa_checklist.py --only sql_total_to_northstar
  python qa/run_qa_checklist.py --skip-judge    # deterministic-only, fast
  python qa/run_qa_checklist.py --raw           # no grading; dump responses
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "qa") not in sys.path:
    sys.path.insert(0, str(ROOT / "qa"))

from checklist_cases import QA_CASES

ARTIFACT_DIR = ROOT / "artifacts" / "qa_runs"
APP_MODULE = None

MODE_ROUTES = {
    "rag": "/api/ask",
    "sql": "/api/sql/query",
    "graph": "/api/graph/query",
    "auto": "/api/auto/query",
}

JUDGE_SYSTEM_PROMPT = """You are a strict QA evaluator for a local banking assistant called Butler.

You will be given a test question, the expected behavior, the assistant's actual
answer, the structured API payload (rows, selected modes, SQL, Cypher), and a
list of deterministic substring hints with whether each was found.

Your job:
1. Decide whether the actual result satisfies the expected behavior.
2. Ground your judgement in the structured payload (rows, selected_modes, SQL,
   Cypher) and the answer text. Treat the substring hints as evidence, not as a
   hard requirement: a missing hint is acceptable if the same fact is clearly
   present in another form.
3. Do not require identical wording when the answer is semantically correct.
4. Fail if key entities, values, or the required refusal/fallback behavior are
   missing, wrong, or contradicted.
5. Return ONLY valid JSON with this exact shape:
   {"verdict":"pass","reason":"Short explanation."}

Use only "pass" or "fail" for verdict.
"""


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def ensure_json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def get_app_module():
    global APP_MODULE
    if APP_MODULE is None:
        APP_MODULE = importlib.import_module("app")
    return APP_MODULE


def extract_answer(mode: str, payload: dict[str, Any]) -> str:
    if mode == "sql":
        return str(payload.get("summary", "")).strip()
    if mode == "graph":
        return str(payload.get("summary", "")).strip()
    # rag + auto
    return str(payload.get("answer", "")).strip()


def build_result_text(mode: str, payload: dict[str, Any]) -> str:
    data = {
        "mode": mode,
        "answer": extract_answer(mode, payload),
        "summary": payload.get("summary", ""),
        "selected_modes": payload.get("selected_modes", []),
        "route": payload.get("route", {}),
        "sql": payload.get("sql", ""),
        "columns": payload.get("columns", []),
        "rows": payload.get("rows", []),
        "cypher": payload.get("cypher", ""),
        "results": payload.get("results", []),
        "sources": payload.get("sources", []),
        "partial_errors": payload.get("partial_errors", []),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Judge did not return a JSON object.")


def run_case(client, case: dict[str, Any]) -> dict[str, Any]:
    route = MODE_ROUTES[case["mode"]]
    try:
        response = client.post(route, json={"question": case["question"]})
    except Exception as exc:  # pragma: no cover - infra failure path
        return {"status_code": 0, "payload": {}, "answer": "", "transport_error": str(exc)}

    payload = response.get_json(silent=True)
    if payload is None:
        payload = {"raw_text": response.get_data(as_text=True)}
    payload = ensure_json_safe(payload)

    return {
        "status_code": response.status_code,
        "payload": payload,
        "answer": extract_answer(case["mode"], payload if isinstance(payload, dict) else {}),
        "transport_error": None,
    }


def evaluate_strict_gates(case: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic checks that independently fail a case."""
    payload = result["payload"]
    answer = result["answer"]
    checks: list[dict[str, Any]] = []

    checks.append({
        "name": "http_status_ok",
        "passed": result["status_code"] == 200,
        "details": f"Expected 200, got {result['status_code']}.",
    })

    exact_answer = case.get("exact_answer")
    if exact_answer is not None:
        checks.append({
            "name": "exact_answer",
            "passed": normalize_text(answer) == normalize_text(exact_answer),
            "details": f"Expected exact answer: {exact_answer!r}; got: {answer!r}",
        })

    actual_modes = payload.get("selected_modes", []) if isinstance(payload, dict) else []

    expected_modes = case.get("expected_modes")
    if expected_modes is not None:
        checks.append({
            "name": "expected_modes",
            "passed": sorted(actual_modes) == sorted(expected_modes),
            "details": f"Expected modes {sorted(expected_modes)}, got {sorted(actual_modes)}.",
        })

    required_modes = case.get("required_modes", [])
    if required_modes:
        checks.append({
            "name": "required_modes_present",
            "passed": all(m in actual_modes for m in required_modes),
            "details": f"Required modes {required_modes}, got {actual_modes}.",
        })

    forbidden_modes = case.get("forbidden_modes", [])
    if forbidden_modes:
        checks.append({
            "name": "forbidden_modes_absent",
            "passed": all(m not in actual_modes for m in forbidden_modes),
            "details": f"Forbidden modes {forbidden_modes}, got {actual_modes}.",
        })

    return checks


def evaluate_substring_hints(case: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    """Advisory substring checks. Fed to the judge as hints; deterministic
    grader when --skip-judge is used."""
    haystack = normalize_text(build_result_text(case["mode"], result["payload"]))
    hints: list[dict[str, Any]] = []

    for text in case.get("must_contain", []):
        hints.append({
            "name": f"contains:{text}",
            "passed": normalize_text(text) in haystack,
            "kind": "must_contain",
        })
    for text in case.get("must_not_contain", []):
        hints.append({
            "name": f"not_contains:{text}",
            "passed": normalize_text(text) not in haystack,
            "kind": "must_not_contain",
        })
    return hints


def evaluate_with_judge(case: dict[str, Any], result: dict[str, Any], hints: list[dict[str, Any]]) -> dict[str, Any]:
    hint_lines = "\n".join(
        f"- {'FOUND' if h['passed'] else 'MISSING'} ({h['kind']}): {h['name'].split(':', 1)[1]}"
        for h in hints
    ) or "(none)"

    prompt = f"""Mode: {case["mode"]}
Question: {case["question"]}
What it tests: {case["what_it_tests"]}
Expected behavior: {case["expected_behavior"]}

Substring hints (evidence only, not hard requirements):
{hint_lines}

Actual answer:
{result["answer"]}

Actual payload:
{build_result_text(case["mode"], result["payload"])}

Return the verdict JSON now.
"""

    last_error = None
    for _ in range(2):
        raw = get_app_module().llm.answer(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            max_tokens=220,
        )
        try:
            parsed = extract_json_object(raw)
        except ValueError as exc:
            last_error = str(exc)
            continue
        verdict = str(parsed.get("verdict", "fail")).strip().lower()
        reason = str(parsed.get("reason", "Judge did not provide a reason.")).strip()
        return {
            "ok": True,
            "passed": verdict == "pass",
            "verdict": verdict,
            "reason": reason,
            "raw": raw,
        }

    return {"ok": False, "passed": False, "verdict": "error", "reason": last_error or "Judge parse failed.", "raw": raw}


def grade_case(case: dict[str, Any], result: dict[str, Any], skip_judge: bool) -> dict[str, Any]:
    strict_gates = evaluate_strict_gates(case, result)
    hints = evaluate_substring_hints(case, result)

    transport_error = result.get("transport_error")
    server_error = result["status_code"] >= 500 or result["status_code"] == 0
    strict_passed = all(c["passed"] for c in strict_gates)

    judge = None
    judge_error = False
    if not skip_judge and not server_error:
        judge = evaluate_with_judge(case, result, hints)
        judge_error = not judge["ok"]

    # Determine content verdict + overall status.
    if transport_error or server_error:
        status = "ERROR"
        passed = False
    elif judge_error:
        status = "ERROR"
        passed = False
    else:
        if skip_judge:
            content_passed = all(h["passed"] for h in hints)
        else:
            content_passed = judge["passed"]
        passed = strict_passed and content_passed
        status = "PASS" if passed else "FAIL"

    return {
        "strict_gates": strict_gates,
        "hints": hints,
        "judge": judge,
        "strict_passed": strict_passed,
        "status": status,
        "passed": passed,
    }


def render_markdown_report(run: dict[str, Any]) -> str:
    lines = [
        "# QA Checklist Report",
        "",
        f"- Timestamp: `{run['timestamp']}`",
        f"- Mode filter: `{run['mode_filter'] or 'all'}`",
        f"- Judge: `{'off' if run['skip_judge'] else 'on'}`",
        f"- Result: **{run['passed_cases']}/{run['total_cases']} passed**, "
        f"{run['failed_cases']} failed, {run['error_cases']} errors",
        "",
        "## By mode",
        "",
        "| Mode | Passed | Total |",
        "| --- | --- | --- |",
    ]
    for mode, stats in run["by_mode"].items():
        lines.append(f"| {mode} | {stats['passed']} | {stats['total']} |")

    lines += ["", "## Case Results", ""]

    for c in run["cases"]:
        lines += [
            f"### {c['status']} — {c['id']}",
            "",
            f"- Mode: `{c['mode']}`",
            f"- Question: {c['question']}",
            f"- Expected: {c['expected_behavior']}",
            f"- HTTP status: `{c['status_code']}`",
            f"- Answer: {c['answer'] or '(empty)'}",
        ]
        if c.get("selected_modes"):
            lines.append(f"- Selected modes: `{', '.join(c['selected_modes'])}`")
        if c.get("sql"):
            lines.append(f"- SQL: `{c['sql']}`")
        if c.get("cypher"):
            lines.append(f"- Cypher: `{c['cypher']}`")
        if c.get("judge"):
            lines.append(f"- Judge: `{c['judge']['verdict']}` — {c['judge']['reason']}")

        failed_gates = [g for g in c["strict_gates"] if not g["passed"]]
        if failed_gates:
            lines.append("- Failed strict gates:")
            for g in failed_gates:
                lines.append(f"  - `{g['name']}`: {g['details']}")

        missing_hints = [h for h in c["hints"] if not h["passed"]]
        if missing_hints:
            lines.append("- Missing substring hints: " + ", ".join(h["name"] for h in missing_hints))

        if c.get("partial_errors"):
            lines.append(f"- Partial errors: `{json.dumps(c['partial_errors'], ensure_ascii=False)}`")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid QA checklist against Butler.")
    parser.add_argument("--mode", choices=list(MODE_ROUTES), help="Run only one mode.")
    parser.add_argument("--only", help="Run only the case with this id.")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip the local-LLM judge; grade with deterministic substring checks only.")
    parser.add_argument("--raw", action="store_true",
                        help="Dump responses with no grading (artifact for manual review).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    selected = [
        c for c in QA_CASES
        if (not args.mode or c["mode"] == args.mode) and (not args.only or c["id"] == args.only)
    ]
    if not selected:
        print("No cases matched the given filters.")
        return 2

    skip_judge = args.skip_judge or args.raw
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = {
        "timestamp": timestamp,
        "mode_filter": args.mode,
        "only": args.only,
        "skip_judge": skip_judge,
        "raw": args.raw,
        "total_cases": len(selected),
        "passed_cases": 0,
        "failed_cases": 0,
        "error_cases": 0,
        "by_mode": {},
        "cases": [],
    }

    app_module = get_app_module()
    with app_module.app.test_client() as client:
        for i, case in enumerate(selected, 1):
            print(f"[{i}/{len(selected)}] {case['mode']:5s} {case['id']} ...", flush=True)
            result = run_case(client, case)

            if args.raw:
                grading = {
                    "strict_gates": [], "hints": evaluate_substring_hints(case, result),
                    "judge": None, "strict_passed": None, "status": "RAW", "passed": None,
                }
            else:
                grading = grade_case(case, result, skip_judge)

            payload = result["payload"] if isinstance(result["payload"], dict) else {}
            run["cases"].append({
                "id": case["id"],
                "mode": case["mode"],
                "question": case["question"],
                "what_it_tests": case["what_it_tests"],
                "expected_behavior": case["expected_behavior"],
                "status_code": result["status_code"],
                "transport_error": result.get("transport_error"),
                "answer": result["answer"],
                "selected_modes": payload.get("selected_modes", []),
                "route": payload.get("route", {}),
                "sql": payload.get("sql", ""),
                "cypher": payload.get("cypher", ""),
                "partial_errors": payload.get("partial_errors", []),
                "payload": payload,
                "strict_gates": grading["strict_gates"],
                "hints": grading["hints"],
                "judge": grading["judge"],
                "status": grading["status"],
                "passed": grading["passed"],
            })

            stats = run["by_mode"].setdefault(case["mode"], {"passed": 0, "total": 0})
            stats["total"] += 1
            if grading["status"] == "PASS":
                run["passed_cases"] += 1
                stats["passed"] += 1
            elif grading["status"] == "ERROR":
                run["error_cases"] += 1
            elif grading["status"] == "FAIL":
                run["failed_cases"] += 1

    json_path = ARTIFACT_DIR / f"{timestamp}.json"
    md_path = ARTIFACT_DIR / f"{timestamp}.md"
    json_path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    md_path.write_text(render_markdown_report(run))

    print(f"\nSaved raw payloads to {json_path}")
    print(f"Saved readable report to {md_path}")
    if args.raw:
        print(f"RAW dump complete: {run['total_cases']} cases recorded (no grading).")
        return 0

    per_mode = " · ".join(f"{m} {s['passed']}/{s['total']}" for m, s in run["by_mode"].items())
    print(f"\nQA result: {run['passed_cases']}/{run['total_cases']} passed, "
          f"{run['failed_cases']} failed, {run['error_cases']} errors.")
    print(f"By mode: {per_mode}")
    return 0 if run["failed_cases"] == 0 and run["error_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
