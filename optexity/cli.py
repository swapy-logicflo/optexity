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
    automation = compile_automation(cache, prune=None if args.no_llm else llm_prune)
    with open(args.output, "w") as f:
        f.write(automation_to_json(automation))
    print(
        f"Compiled {len(cache.actions)} cached actions into {len(automation.nodes)} nodes -> {args.output}"
    )


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
    compile_cmd.set_defaults(func=compile_cache)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
