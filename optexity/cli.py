import argparse
import logging
import os
import subprocess
import sys

from dotenv import load_dotenv
from uvicorn import run

logger = logging.getLogger(__name__)

env_path = os.getenv("ENV_PATH")
if not env_path:
    logger.warning("ENV_PATH is not set, using default values")
else:
    load_dotenv(env_path)


def install_browsers() -> None:
    """Install Playwright + Patchright browsers."""
    try:
        subprocess.run(
            ["playwright", "install", "--with-deps", "chromium", "chrome"],
            check=True,
        )
        subprocess.run(
            ["patchright", "install", "chromium", "chrome"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install browsers", file=sys.stderr)
        sys.exit(e.returncode)


def run_inference(args: argparse.Namespace) -> None:
    from optexity.inference.child_process import get_app_with_endpoints

    app = get_app_with_endpoints(
        is_aws=args.is_aws, child_id=args.child_process_id, port=args.port
    )
    run(
        app,
        host=args.host,
        port=args.port,
    )


def compile_cache(args: argparse.Namespace) -> None:
    from browser_use.agent.action_cache import ActionCache

    from optexity.learning.compiler import automation_to_json, compile_automation
    from optexity.learning.pruner import llm_prune

    cache = ActionCache.load(args.cache_path)
    automation = compile_automation(
        cache, prune=None if args.no_llm else llm_prune, parameterize=args.parameterize
    )
    with open(args.output, "w") as f:
        f.write(automation_to_json(automation))
    print(
        f"Compiled {len(cache.actions)} cached actions into {len(automation.nodes)} nodes -> {args.output}"
    )


def run_learning_loop(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from optexity.learning.compiler import automation_to_json
    from optexity.learning.loop import InferenceServer, learn
    from optexity.learning.pruner import llm_prune
    from optexity.schema.automation import Automation

    automation = Automation.model_validate_json(Path(args.automation).read_text())
    server = InferenceServer(
        url=args.server,
        endpoint_name=args.endpoint,
        input_parameters=json.loads(args.input_parameters),
        automation_file=Path(args.automation_file),
    )
    output = Path(args.output)

    def report(iteration, ran, result) -> None:
        output.with_suffix(f".iter{iteration}.json").write_text(automation_to_json(ran))
        seconds = f"{result.seconds:.1f}s" if result.seconds is not None else "?"
        print(
            f"iteration {iteration}: {len(ran.nodes)} nodes, status={result.status}, "
            f"llm_calls={result.llm_calls}, tokens={result.tokens}, run={seconds}, task={result.task_id}"
        )

    final, results = learn(
        automation,
        server.run,
        prune=None if args.no_llm else llm_prune,
        max_iterations=args.max_iterations,
        on_iteration=report,
    )
    output.write_text(automation_to_json(final))
    last = results[-1]
    if last.status == "success" and last.llm_calls == 0:
        print(f"Converged: {output} runs with no LLM calls")
    else:
        print(f"Stopped after {len(results)} iterations without converging; latest automation in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="optexity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------------------------
    # install-browsers
    # ---------------------------
    install_cmd = subparsers.add_parser(
        "install_browsers",
        help="Install required browsers for Optexity",
        aliases=["install-browsers"],
    )
    install_cmd.set_defaults(func=lambda _: install_browsers())

    # ---------------------------
    # inference
    # ---------------------------
    inference_cmd = subparsers.add_parser(
        "inference", help="Run Optexity inference server"
    )
    inference_cmd.add_argument("--host", default="0.0.0.0")
    inference_cmd.add_argument("--port", type=int, default=9000)
    inference_cmd.add_argument(
        "--child_process_id", "--child-process-id", type=int, default=0
    )
    inference_cmd.add_argument(
        "--is_aws", "--is-aws", action="store_true", default=False
    )

    inference_cmd.set_defaults(func=run_inference)

    # ---------------------------
    # compile-cache
    # ---------------------------
    compile_cmd = subparsers.add_parser(
        "compile_cache",
        help="Compile an agentic run's action cache into a deterministic automation",
        aliases=["compile-cache"],
    )
    compile_cmd.add_argument("cache_path", help="action_cache.json written by an agentic task")
    compile_cmd.add_argument("-o", "--output", default="test_automation_cached.json")
    compile_cmd.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM pass that removes semantically redundant actions",
    )
    compile_cmd.add_argument(
        "--parameterize",
        action="store_true",
        help="Turn typed values into input parameters that default to the recorded values",
    )
    compile_cmd.set_defaults(func=compile_cache)

    # ---------------------------
    # learn
    # ---------------------------
    learn_cmd = subparsers.add_parser(
        "learn",
        help="Run, cache and recompile an automation until it needs no LLM calls",
    )
    learn_cmd.add_argument("automation", help="Starting automation JSON, e.g. a single agentic_task")
    learn_cmd.add_argument("--endpoint", required=True, help="Any endpoint on the dashboard; its automation is replaced locally")
    learn_cmd.add_argument("--input-parameters", default="{}", help="JSON matching that endpoint's parameters")
    learn_cmd.add_argument("--server", default="http://localhost:9000")
    learn_cmd.add_argument(
        "--automation-file",
        default="learning_automation.json",
        help="File the server reads; start it with OPTEXITY_LOCAL_AUTOMATION set to this path",
    )
    learn_cmd.add_argument("--max-iterations", type=int, default=4)
    learn_cmd.add_argument("-o", "--output", default="learned_automation.json")
    learn_cmd.add_argument("--no-llm", action="store_true", help="Skip LLM pruning when compiling")
    learn_cmd.set_defaults(func=run_learning_loop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
