"""Zero-dependency constants for the one approved RunPod launch contract.

This module intentionally imports only the Python standard library.  The
watchdog and recovery paths must be importable on a fresh provider image before
Pydantic, PyYAML, or the project virtual environment has been installed.
"""

from __future__ import annotations

LIFECYCLE_PROTOCOL = "runpod-pod-lifecycle-v1"
LIFECYCLE_STATE_FILENAME = "pod_lifecycle.json"
GPU_COMMAND_PHASES = frozenset(
    {
        "behavior_baseline_gpu",
        "behavior_treatment_gpu",
        "resample_gpu",
        "lens_gpu",
    }
)
EXACT_PROVIDER_GPU_ID = "NVIDIA H100 80GB HBM3"
EXACT_GPU_FAMILY = "H100_80GB"
EXACT_GPU_COUNT = 8
EXACT_CLOUD = "SECURE"
EXACT_CUDA_VERSIONS = ("12.8",)
CANDIDATE_DATA_CENTER_IDS = frozenset({"CA-MTL-1", "EUR-IS-3"})
EXACT_CONTAINER_DISK_GB = 50
EXACT_VOLUME_DISK_GB = 650
EXACT_VOLUME_MOUNT_PATH = "/workspace"
EXACT_PORTS = ("22/tcp",)

SESSION_ENV_NAME = "GPU_BUDGET_SESSION_ID"
HF_TOKEN_ENV_NAME = "HF_TOKEN"
STATIC_POD_ENV = {
    "HF_HOME": "/workspace/.cache/huggingface",
    "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
    "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
    "VLLM_CACHE_ROOT": "/workspace/.cache/vllm",
    "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
}
REQUESTED_POD_ENV_KEYS = frozenset({HF_TOKEN_ENV_NAME, SESSION_ENV_NAME, *STATIC_POD_ENV})
PROVIDER_MANAGED_ENV_KEYS = frozenset({"PUBLIC_KEY"})
TERMINAL_POD_STATUSES = frozenset({"EXITED", "TERMINATED"})

__all__ = [
    "CANDIDATE_DATA_CENTER_IDS",
    "EXACT_CLOUD",
    "EXACT_CONTAINER_DISK_GB",
    "EXACT_CUDA_VERSIONS",
    "EXACT_GPU_COUNT",
    "EXACT_GPU_FAMILY",
    "EXACT_PORTS",
    "EXACT_PROVIDER_GPU_ID",
    "EXACT_VOLUME_DISK_GB",
    "EXACT_VOLUME_MOUNT_PATH",
    "GPU_COMMAND_PHASES",
    "HF_TOKEN_ENV_NAME",
    "LIFECYCLE_PROTOCOL",
    "LIFECYCLE_STATE_FILENAME",
    "PROVIDER_MANAGED_ENV_KEYS",
    "REQUESTED_POD_ENV_KEYS",
    "SESSION_ENV_NAME",
    "STATIC_POD_ENV",
    "TERMINAL_POD_STATUSES",
]
