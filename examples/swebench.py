#!/usr/bin/env python3
"""
Simple Sandbox API Demo - shows auth and basic usage
"""

# from swebench.harness.constants import (
#     APPLY_PATCH_FAIL,
#     APPLY_PATCH_PASS,
#     DOCKER_PATCH,
#     DOCKER_USER,
#     DOCKER_WORKDIR,
#     INSTANCE_IMAGE_BUILD_DIR,
#     KEY_INSTANCE_ID,
#     KEY_MODEL,
#     KEY_PREDICTION,
#     LOG_REPORT,
#     LOG_INSTANCE,
#     LOG_TEST_OUTPUT,
#     RUN_EVALUATION_LOG_DIR,
#     UTF8,
# )
exit()

from prime_cli.api.client import APIClient, APIError
from prime_cli.api.sandbox import CreateSandboxRequest, SandboxClient


def run_instance(sandbox_client: SandboxClient, pred: dict) -> None:
    """
    Run a single instance with the given prediction.

    Args:
        sandbox_client (SandboxClient): Sandbox client
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
    """


def main() -> None:
    """Simple sandbox demo"""
    try:
        # 1. Authentication - uses API key from config or environment
        # Run 'prime login' first to set up your API key
        client = APIClient()  # Automatically loads API key from ~/.prime/config.json
        sandbox_client = SandboxClient(client)

        # 2. Create a sandbox
        request = CreateSandboxRequest(
            name="sympy__sympy-21596",
            docker_image="swebench/sweb.eval.x86_64.sympy_1776_sympy-21596",
            start_command="tail -f /dev/null",  # Keep container running indefinitely
            cpu_cores=1,
            memory_gb=2,
            timeout_minutes=120,  # 2 hours to avoid timeout during demo
        )

        print("Creating sandbox...")
        sandbox = sandbox_client.create(request)
        print(f"✅ Created: {sandbox.name} ({sandbox.id})")

        # 3. Wait for sandbox to be running
        import time

        print("\nWaiting for sandbox to be running...")
        max_attempts = 30
        for _ in range(max_attempts):
            sandbox = sandbox_client.get(sandbox.id)
            if sandbox.status == "RUNNING":
                print("✅ Sandbox is running!")
                # Give it a few extra seconds to be ready for commands
                time.sleep(10)
                break
            elif sandbox.status in ["ERROR", "TERMINATED"]:
                print(f"❌ Sandbox failed with status: {sandbox.status}")
                return
            time.sleep(2)

        # 4. Run a command
        result = sandbox_client.execute_command(sandbox.id, "pwd")
        print(f"Working directory: {result.stdout.strip()}")

        # 7. Clean up
        print(f"\nDeleting {sandbox.name}...")
        sandbox_client.delete(sandbox.id)
        print("✅ Deleted")

    except APIError as e:
        print(f"❌ API Error: {e}")
        print("💡 Make sure you're logged in: run 'prime login' first")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
