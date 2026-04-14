"""Shared preprocessing constants used by dataset build and runtime inference."""

from __future__ import annotations

SEQUENCE_LENGTH = 90
FULL_BODY25_JOINTS = 25
SELECTED_BODY25_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18]
NUM_JOINTS = len(SELECTED_BODY25_INDICES)
NUM_COORDS = 2  # x, y only for model v1
FEATURES_PER_FRAME = NUM_JOINTS * NUM_COORDS

# Selected upper-body BODY_25 subset indices (local subset order)
# [Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist,
#  MidHip, RHip, LHip, REye, LEye, REar, LEar]
NECK_IDX = 1
R_SHOULDER_IDX = 2
L_SHOULDER_IDX = 5
MID_HIP_IDX = 8
R_HIP_IDX = 9
L_HIP_IDX = 10

# Confidence handling (weak filter)
DEFAULT_CONFIDENCE_CUTOFF = 0.05

# Weighted robust scale candidates
SHOULDER_WEIGHT = 0.50
TORSO_WEIGHT = 0.35
HIP_WEIGHT = 0.15

# Safety / smoothing constants
MIN_SCALE_EPS = 1e-3
SAFE_FALLBACK_SCALE = 100.0
SCALE_SMOOTH_ALPHA_OLD = 0.8
SCALE_SMOOTH_ALPHA_NEW = 0.2

# Suspicious frame thresholds
STABLE_JOINTS = [NECK_IDX, L_SHOULDER_IDX, R_SHOULDER_IDX, MID_HIP_IDX]
MULTI_JOINT_JUMP_SCALE_MULTIPLIER = 1.25
CENTER_JUMP_SCALE_MULTIPLIER = 1.50
MULTI_JOINT_MIN_COUNT = 2

# Upper-body subset left/right symmetry pairs for joint-level fallback imputation.
# Each tuple is (left_joint_index, right_joint_index).
SYMMETRIC_JOINT_PAIRS: list[tuple[int, int]] = [
    (2, 5),   # RShoulder <-> LShoulder
    (3, 6),   # RElbow <-> LElbow
    (4, 7),   # RWrist <-> LWrist
    (9, 10),  # RHip <-> LHip
    (11, 12), # REye <-> LEye
    (13, 14), # REar <-> LEar
]
