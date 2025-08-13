#!/usr/bin/env python3
"""
adapted from https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/run_evaluation.py
"""
from __future__ import annotations

import docker
import json
import platform
import traceback

if platform.system() == "Linux":
    import resource

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path, PurePosixPath

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from swebench.harness.docker_utils import (
    clean_images,
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    list_images,
    remove_image,
    should_remove,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.reporting import make_run_report
from swebench.harness.modal_eval import (
    run_instances_modal,
    validate_modal_credentials,
)
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import (
    EvaluationError,
    load_swebench_dataset,
    get_predictions_from_file,
    run_threadpool,
    str2bool,
    optional_str,
)

from prime_cli.api.client import APIClient, APIError
from prime_cli.api.sandbox import CommandResponse, CreateSandboxRequest, SandboxClient

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def pipe_file_content_into_sandbox(
    sandbox_client: SandboxClient,
    sandbox_id: str,
    file_path: str,
    content: str
) -> CommandResponse:
    return sandbox_client.execute_command(sandbox_id, f"echo '{content}' > {file_path}")


def run_instance(
    test_spec: TestSpec,
    pred: dict,
    sandbox_client: SandboxClient,
    run_id: str,
    timeout: int | None = None,
    rewrite_reports: bool = False,
):
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
        rewrite_reports (bool): True if eval run is just to reformat existing report
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id

    # Set up report file
    report_path = log_dir / LOG_REPORT
    if rewrite_reports:
        test_output_path = log_dir / LOG_TEST_OUTPUT
        if not test_output_path.exists():
            raise ValueError(f"Test output file {test_output_path} does not exist")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())

    if not test_spec.is_remote_image:
        # Link the image build dir in the log dir
        build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(
            ":", "__"
        )
        image_build_link = log_dir / "image_build_dir"
        if not image_build_link.exists():
            try:
                # link the image build dir in the log dir
                image_build_link.symlink_to(
                    build_dir.absolute(), target_is_directory=True
                )
            except:
                # some error, idk why
                pass

    # Set up logger
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    sandbox = None
    try:
        # Build + start instance container (instance image should already be built)
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name=f"swebench-{instance_id}",
                docker_image=test_spec.instance_image_key,
                start_command="tail -f /dev/null",
                cpu_cores=1,
                memory_gb=2,
                timeout_minutes=120,  # 2 hours to avoid timeout during demo
                # working_dir=DOCKER_WORKDIR,
            )
        )
        sandbox_client.wait_for_sandbox(sandbox.id, max_attempts=60)
        logger.info(f"Sandbox for {instance_id} started: {sandbox.id}")
        cmd_response = sandbox_client.execute_command(
            sandbox.id,
            "git config --global --add safe.directory /testbed",
            # working_dir=DOCKER_WORKDIR,
        )
        logger.info(f"git config --global --add safe.directory /testbed: stdout: {cmd_response.stdout}\nstderr: {cmd_response.stderr}")
        pwd_response = sandbox_client.execute_command(
            sandbox.id,
            "pwd",
            working_dir=DOCKER_WORKDIR,
        )
        logger.info(f"pwd: stdout: {pwd_response.stdout}\nstderr: {pwd_response.stderr}")
        ls_response = sandbox_client.execute_command(
            sandbox.id,
            "ls -lah",
            working_dir=DOCKER_WORKDIR,
        )
        logger.info(f"ls -lah: stdout: {ls_response.stdout}\nstderr: {ls_response.stderr}")
        ls_response = sandbox_client.execute_command(
            sandbox.id,
            "ls -lah .git/",
            working_dir=DOCKER_WORKDIR,
        )
        logger.info(f"ls -lah: stdout: {ls_response.stdout}\nstderr: {ls_response.stderr}")
        git_status_response = sandbox_client.execute_command(
            sandbox.id,
            "git -C /testbed status",
            # working_dir=DOCKER_WORKDIR,
        )
        logger.info(f"git status: stdout: {git_status_response.stdout}\nstderr: {git_status_response.stderr}")
        git_rev_parse_response = sandbox_client.execute_command(
            sandbox.id,
            "git -C /testbed rev-parse --git-dir",
            # working_dir=DOCKER_WORKDIR,
        )
        logger.info(f"git -C /testbed rev-parse --git-dir: stdout: {git_rev_parse_response.stdout}\nstderr: {git_rev_parse_response.stderr}")
        whoami_response = sandbox_client.execute_command(
            sandbox.id,
            "whoami",
            # working_dir=DOCKER_WORKDIR,
        )
        logger.info(f"whoami: stdout: {whoami_response.stdout}\nstderr: {whoami_response.stderr}")
        # raise Exception("Stop here")

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred[KEY_PREDICTION] or "")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        # pipe predicted patch into sandbox
        cmd_response = pipe_file_content_into_sandbox(
            sandbox_client=sandbox_client,
            sandbox_id=sandbox.id,
            file_path=DOCKER_PATCH,
            content=patch_file.read_text(),
        )
        logger.info(f"pipe_file_content_into_sandbox:\nstdout: {cmd_response.stdout}\nstderr: {cmd_response.stderr}")

        # Attempt to apply patch to container (TODO: FIX THIS)
        # breakpoint()
        applied_patch = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            pwd_response = sandbox_client.execute_command(
                sandbox.id,
                "pwd",
                working_dir=DOCKER_WORKDIR,
            )
            logger.info(f"pwd: {pwd_response.stdout}")
            ls_response = sandbox_client.execute_command(
                sandbox.id,
                "ls -lah",
                working_dir=DOCKER_WORKDIR,
            )
            logger.info(f"ls -lah: {ls_response.stdout}")
            cmd_response = sandbox_client.execute_command(
                sandbox.id,
                f"{git_apply_cmd} {DOCKER_PATCH}",
                working_dir=DOCKER_WORKDIR,
            )
            if cmd_response.exit_code == 0:
                logger.info(f"{APPLY_PATCH_PASS}:\n{cmd_response.stdout}")
                applied_patch = True
                break
            else:
                logger.info(f"Failed to apply patch to container: {git_apply_cmd}\nstdout: {cmd_response.stdout}\nstderr: {cmd_response.stderr}")
        if not applied_patch:
            logger.info(f"{APPLY_PATCH_FAIL}:\nstdout: {cmd_response.stdout}\nstderr: {cmd_response.stderr}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\nstdout: {cmd_response.stdout}\nstderr: {cmd_response.stderr}",
                logger,
            )

        cmd_response = (
            sandbox_client.execute_command(
                sandbox.id,
                "pwd",
                working_dir=DOCKER_WORKDIR,
            )
        )
        logger.info(f"pwd: {cmd_response.stdout}")
        cmd_response = (
            sandbox_client.execute_command(
                sandbox.id,
                "ls -lah",
                working_dir=DOCKER_WORKDIR,
            )
        )
        logger.info(f"ls -lah: {cmd_response.stdout}")
        # Get git diff before running eval script
        cmd_response = (
            sandbox_client.execute_command(
                sandbox.id,
                "git -c core.fileMode=false diff",
                working_dir=DOCKER_WORKDIR,
            )
        )
        git_diff_output_before = cmd_response.stdout
        logger.info(f"Git diff before:\n{git_diff_output_before}")
        logger.info(f"\nstderr: {cmd_response.stderr}")

        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}; copying to container..."
        )
        # pipe eval script into sandbox
        pipe_file_content_into_sandbox(
            sandbox_client=sandbox_client,
            sandbox_id=sandbox.id,
            file_path=PurePosixPath("/eval.sh"),
            content=eval_file.read_text(),
        )

        # Run eval script, write output to logs
        cmd_response = sandbox_client.execute_command(
            sandbox_id=sandbox.id,
            command="/bin/bash /eval.sh",
            working_dir=DOCKER_WORKDIR,
        )
        test_output_path = log_dir / LOG_TEST_OUTPUT
        test_output = cmd_response.stdout
        with open(test_output_path, "w") as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")

        # Get git diff after running eval script (ignore permission changes)
        cmd_response = (
            sandbox_client.execute_command(
                sandbox_id=sandbox.id,
                command="git -c core.fileMode=false diff",
                working_dir=DOCKER_WORKDIR,
            )
        )
        git_diff_output_after = cmd_response.stdout

        # Check if git diff changed after running eval script
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        logger.info(f"\nstderr: {cmd_response.stderr}")
        if git_diff_output_after != git_diff_output_before:
            logger.info("Git diff changed after running eval script")

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )

        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except APIError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
        logger.error(error_msg)
    finally:
        # Remove instance container + image, close logger
        sandbox_client.delete(sandbox.id)
        close_logger(logger)
    return


def run_instances(
    predictions: dict,
    instances: list,
    cache_level: str,
    clean: bool,
    force_rebuild: bool,
    max_workers: int,
    run_id: str,
    timeout: int,
    namespace: str | None = "swebench",
    instance_image_tag: str = "latest",
    rewrite_reports: bool = False,
):
    """
    Run all instances for the given predictions in parallel.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    # client = docker.from_env()
    client = APIClient()  # Automatically loads API key from ~/.prime/config.json
    sandbox_client = SandboxClient(client)
    test_specs = list(
        map(
            lambda instance: make_test_spec(
                instance, namespace=namespace, instance_image_tag=instance_image_tag
            ),
            instances,
        )
    )

    # print number of existing instance images
    # instance_image_ids = {x.instance_image_key for x in test_specs}
    # existing_images = {
    #     tag
    #     for i in client.images.list(all=True)
    #     for tag in i.tags
    #     if tag in instance_image_ids
    # }
    # if not force_rebuild and len(existing_images):
    #     print(
    #         f"Found {len(existing_images)} existing instance images. Will reuse them."
    #     )

    # run instances in parallel
    payloads = []
    for test_spec in test_specs:
        payloads.append(
            (
                test_spec,
                predictions[test_spec.instance_id],
                sandbox_client,
                run_id,
                timeout,
                rewrite_reports,
            )
        )

    # run instances in parallel
    print(f"Running {len(instances)} instances...")
    # run_threadpool(run_instance, payloads, max_workers)
    # run_instance(*payloads[0])
    run_instance(*payloads[1])
    print("All instances run.")


def get_dataset_from_preds(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions: dict,
    run_id: str,
    rewrite_reports: bool,
    exclude_completed: bool = True,
):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    # load dataset
    dataset = load_swebench_dataset(dataset_name, split)
    dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}

    if instance_ids:
        # check that all instance IDs have predictions
        missing_preds = set(instance_ids) - set(predictions.keys())
        if missing_preds:
            print(
                f"Warning: Missing predictions for {len(missing_preds)} instance IDs."
            )

    # check that all prediction IDs are in the dataset
    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        raise ValueError(
            (
                "Some prediction IDs not found in dataset!"
                f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
            )
        )
    if instance_ids:
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]

    if rewrite_reports:
        # we only return instances that have existing test outputs
        test_output_ids = set()
        for instance in dataset:
            if instance[KEY_INSTANCE_ID] not in predictions:
                continue
            prediction = predictions[instance[KEY_INSTANCE_ID]]
            test_output_file = (
                RUN_EVALUATION_LOG_DIR
                / run_id
                / prediction["model_name_or_path"].replace("/", "__")
                / prediction[KEY_INSTANCE_ID]
                / "test_output.txt"
            )
            if test_output_file.exists():
                test_output_ids.add(instance[KEY_INSTANCE_ID])
        dataset = [
            i
            for i in dataset
            if i[KEY_INSTANCE_ID] in prediction_ids
            and i[KEY_INSTANCE_ID] in test_output_ids
        ]
        return dataset

    # check which instance IDs have already been run
    completed_ids = set()
    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in prediction_ids:
            # skip instances without predictions
            continue
        prediction = predictions[instance[KEY_INSTANCE_ID]]
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction[KEY_MODEL].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
            / LOG_REPORT
        )
        if report_file.exists():
            completed_ids.add(instance[KEY_INSTANCE_ID])

    if completed_ids and exclude_completed:
        # filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]

    empty_patch_ids = {
        k
        for k, v in predictions.items()
        if v[KEY_PREDICTION] == "" or v[KEY_PREDICTION] is None
    }

    # filter dataset to only instances with predictions
    dataset = [
        i
        for i in dataset
        if i[KEY_INSTANCE_ID] in prediction_ids
        and i[KEY_INSTANCE_ID] not in empty_patch_ids
    ]
    return dataset


def main(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions_path: str,
    max_workers: int,
    force_rebuild: bool,
    cache_level: str,
    clean: bool,
    open_file_limit: int,
    run_id: str,
    timeout: int,
    namespace: str | None,
    rewrite_reports: bool,
    modal: bool,
    instance_image_tag: str = "latest",
    report_dir: str = ".",
):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    if dataset_name == "SWE-bench/SWE-bench_Multimodal" and split == "test":
        print(
            "⚠️ Local evaluation for the test split of SWE-bench Multimodal is not supported. "
            "Please check out sb-cli (https://github.com/swe-bench/sb-cli/) for instructions on how to submit predictions."
        )
        return

    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    if report_dir is not None:
        report_dir = Path(report_dir)
        if not report_dir.exists():
            report_dir.mkdir(parents=True)

    if force_rebuild and namespace is not None:
        raise ValueError("Cannot force rebuild and use a namespace at the same time.")

    # load predictions as map of instance_id to prediction
    predictions = get_predictions_from_file(predictions_path, dataset_name, split)
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # get dataset from predictions
    dataset = get_dataset_from_preds(
        dataset_name, split, instance_ids, predictions, run_id, rewrite_reports
    )
    full_dataset = load_swebench_dataset(dataset_name, split, instance_ids)

    if modal:
        # run instances on Modal
        if not dataset:
            print("No instances to run.")
        else:
            validate_modal_credentials()
            run_instances_modal(predictions, dataset, full_dataset, run_id, timeout)
        return

    # run instances locally
    if platform.system() == "Linux":
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    existing_images = list_images(client)
    if not dataset:
        print("No instances to run.")
    else:
        # build environment images + run instances
        if namespace is None and not rewrite_reports:
            build_env_images(client, dataset, force_rebuild, max_workers)
        run_instances(
            predictions,
            dataset,
            cache_level,
            clean,
            force_rebuild,
            max_workers,
            run_id,
            timeout,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            rewrite_reports=rewrite_reports,
        )

    # clean images + make final report
    clean_images(client, existing_images, cache_level, clean)
    return make_run_report(predictions, full_dataset, run_id, client)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run evaluation harness for the given dataset and predictions.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Common args
    parser.add_argument(
        "--dataset_name",
        default="SWE-bench/SWE-bench_Lite",
        type=str,
        help="Name of dataset or path to JSON file.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split of the dataset"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        help="Path to predictions file - if 'gold', uses gold predictions",
        required=True,
    )

    # Local execution args
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of workers (should be <= 75%% of CPU cores)",
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=4096, help="Open file limit"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout (in seconds) for running tests for each instance",
    )
    parser.add_argument(
        "--force_rebuild",
        type=str2bool,
        default=False,
        help="Force rebuild of all images",
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID - identifies the run"
    )
    parser.add_argument(
        "--namespace",
        type=optional_str,
        default="swebench",
        help='Namespace for images. (use "none" to use no namespace)',
    )
    parser.add_argument(
        "--instance_image_tag", type=str, default="latest", help="Instance image tag"
    )
    parser.add_argument(
        "--rewrite_reports",
        type=str2bool,
        default=False,
        help="Doesn't run new instances, only writes reports for instances with existing test outputs",
    )
    parser.add_argument(
        "--report_dir", type=str, default=".", help="Directory to write reports to"
    )

    # Modal execution args
    parser.add_argument("--modal", type=str2bool, default=False, help="Run on Modal")

    args = parser.parse_args()
    main(**vars(args))
