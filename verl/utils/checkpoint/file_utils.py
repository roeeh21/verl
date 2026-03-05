# Copyright 2025 AI21 Labs

import logging
import os
import subprocess

from verl.utils.py_functional import threaded

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def sync_paths_gs(source_path: str, dest_path: str, wildcard: str, log_stdout: bool = False, fast_timeout: int = 150):
    try:
        logger.info(f"Syncing paths from {source_path}/{wildcard} to {dest_path}")
        _sync_paths_gs_fast(source_path, dest_path, wildcard, timeout=fast_timeout, log_stdout=log_stdout)
    except subprocess.CalledProcessError as e:
        logger.info(f"sync_paths_gs_fast from {source_path} to {dest_path} failed with exception: {e}")
        logger.info("Falling back to sync_paths_gs_slow")
        _sync_paths_gs_slow(source_path, dest_path, wildcard, log_stdout=log_stdout)


def sync_file_gs(source_path: str, dest_path: str, log_stdout: bool = False):
    logger.info(f"Syncing file from {source_path} to {dest_path}")
    cmd = [
        "gcloud",
        "storage",
        "cp",
        source_path,
        dest_path,
    ]

    _run_cmd(cmd, log_stdout=log_stdout)


@threaded
def sync_paths_gs_threaded(
    source_path: str, dest_path: str, wildcard: str, log_stdout: bool = False, fast_timeout: int = 150
):
    sync_paths_gs(source_path, dest_path, wildcard, log_stdout=log_stdout, fast_timeout=fast_timeout)


@threaded
def sync_file_gs_threaded(source_path: str, dest_path: str, log_stdout: bool = False):
    sync_file_gs(source_path, dest_path, log_stdout=log_stdout)


def check_file_exists_gs(path: str) -> bool:
    cmd = ["gsutil", "stat", path]
    try:
        _run_cmd(cmd, log_stdout=False)
        return True
    except subprocess.CalledProcessError:
        return False


def _sync_paths_gs_slow(source_path: str, dest_path: str, wildcard: str, log_stdout: bool):
    cmd = [
        "gsutil",
        "-m",
        "-o",
        "'GSUtil:parallel_thread_count=1'",
        "-o",
        "'GSUtil:sliced_object_download_max_components=8'",
        "cp",
        f"{source_path}/{wildcard}",
        f"{dest_path}",
    ]

    _run_cmd(cmd, log_stdout=log_stdout)


def _sync_paths_gs_fast(source_path: str, dest_path: str, wildcard: str, log_stdout: bool, timeout=None):
    cmd = ["gsutil", "-m", "cp", "-r", f"{source_path}/{wildcard}", f"{dest_path}"]
    if timeout:
        cmd = ["timeout", f"{timeout}s"] + cmd

    _run_cmd(cmd, log_stdout=log_stdout)


def _run_cmd(cmd: list[str], log_stdout: bool):
    stdout = None
    stderr = None
    shell = False
    if not log_stdout:
        cmd = " ".join(cmd) + " > /dev/null 2>&1"
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        shell = True

    subprocess.check_call(cmd, stdout=stdout, shell=shell, stderr=stderr)
