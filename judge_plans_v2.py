"""Plan judging protocol 2: one call per rank-1 Plan with a 3-4B Hugging Face judge.

Usage:
  python judge_plans_v2.py judge --results RESULTS.jsonl --variants llm_only [...] --out-dir OUT --model HF_ID
  python judge_plans_v2.py validate --model HF_ID --out-dir REPORTS [--cases CASES.jsonl]
  python judge_plans_v2.py compare --reports REPORTS
  python judge_plans_v2.py --version

`judge` never modifies its inputs. It writes OUT/<run-name>/<variant>/manifest.json
and judgments.jsonl (plus judgments.csv with --csv) and resumes from them.
Exit status: 0 complete; 1 unusable inputs/outputs; 2 some Plans still unjudged
(rerun to retry); 3 another writer holds a variant; 130 interrupted.
See med_copilot/docs/plan_judging_v2.md. Protocol 1 remains judge_treatment_plans.py.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.plan_judge_v2 import protocol, runner, validate_models  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["--version"], ["-V"]):
        print(f"{protocol.PROTOCOL_NAME} protocol {protocol.PROTOCOL_VERSION} "
              f"(prompt {protocol.PROMPT_TEMPLATE_SHA256[:12]})")
        return 0
    if argv[:1] == ["judge"]:
        return runner.run(argv[1:])
    parser = argparse.ArgumentParser(prog="judge_plans_v2.py", description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("judge", help="Judge rank-1 Plans (see `judge --help`)", add_help=False)
    validate_models.build_parsers(subparsers)
    args = parser.parse_args(argv)
    if args.command == "validate":
        return validate_models.validate(args)
    return validate_models.compare(args)


if __name__ == "__main__":
    sys.exit(main())
