"""Minimal nuScenes reading (no devkit). Dataset path comes from NUSCENES_ROOT (env var or ./.env)."""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
DRIVING_PROMPT = (
    "These are the 6 surround-view cameras of an ego vehicle, in order: "
    + ", ".join(CAMS)
    + ". Describe the driving scene and what the ego vehicle should do next."
)
_REPO = Path(__file__).resolve().parents[3]


def nuscenes_root() -> Path:
    root = os.environ.get("NUSCENES_ROOT")
    env_file = _REPO / ".env"
    if not root and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("NUSCENES_ROOT="):
                root = line.split("=", 1)[1].strip()
    if not root:
        raise SystemExit("Set NUSCENES_ROOT in the environment or in .env")
    return Path(root)


def load_json(root: Path, version: str, table: str):
    return json.loads((root / version / f"{table}.json").read_text())


def keyframe_index(root: Path, version: str, sample_tokens: set[str]) -> dict[str, dict]:
    """For each wanted sample: {'cams': {CAM: path}, 'calib': {CAM: calibrated_sensor token},
    'ego_pose_token': LIDAR_TOP pose}.

    Scans sample_data.json (1.3 GB for trainval), so call once with all tokens you need.
    """
    index = {t: {"cams": {}, "calib": {}} for t in sample_tokens}
    for sd in load_json(root, version, "sample_data"):
        if not sd["is_key_frame"] or sd["sample_token"] not in index:
            continue
        sensor = sd["filename"].split("/")[1]
        if sensor.startswith("CAM_"):
            index[sd["sample_token"]]["cams"][sensor] = root / sd["filename"]
            index[sd["sample_token"]]["calib"][sensor] = sd["calibrated_sensor_token"]
        elif sensor == "LIDAR_TOP":
            index[sd["sample_token"]]["ego_pose_token"] = sd["ego_pose_token"]
    return index


def scene_sample_tokens(samples: dict, scene: dict) -> list[str]:
    tokens, tok = [], scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = samples[tok]["next"]
    return tokens


def quat_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def to_ego_frame(global_xy: np.ndarray, pose: dict) -> np.ndarray:
    """Global (N, 2) xy -> ego frame of `pose`: x forward, y left, meters."""
    yaw = quat_yaw(pose["rotation"])
    c, s = np.cos(yaw), np.sin(yaw)
    return (global_xy - np.asarray(pose["translation"][:2])) @ np.array([[c, -s], [s, c]])


def load_cams(paths, width=800) -> list[Image.Image]:
    """Load camera images resized to `width` (native 1600x900) to cut vision tokens."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    return [im.resize((width, round(im.height * width / im.width))) for im in ims]
