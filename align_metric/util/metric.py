# align_metric/util/metric.py
import numpy as np
from typing import Dict, Tuple, Optional
import cv2


# ---------------------------
# Basic helpers
# ---------------------------
def mask_to_points(mask01: np.ndarray) -> np.ndarray:
    """
    mask01: (H,W) uint8/bool
    return: (N,2) float32 in (x,y)
    """
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def endpoint_midpoint_by_percentile(
    pts_xy: np.ndarray, which: str = "top", pct: float = 20.0
) -> Optional[np.ndarray]:
    """
    top/bottom percentile band의 평균점을 midpoint로 사용
    """
    if pts_xy.shape[0] == 0:
        return None
    ys = pts_xy[:, 1]
    if which == "top":
        region = pts_xy[ys <= np.percentile(ys, pct)]
    else:
        region = pts_xy[ys >= np.percentile(ys, 100.0 - pct)]
    if region.shape[0] == 0:
        region = pts_xy
    return region.mean(axis=0).astype(np.float32)


def joint_line_from_region(
    pts_xy: np.ndarray, which: str = "bottom", pct: float = 15.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    percentile band에서 x_min~x_max, y_mean으로 '수평 joint line' 생성
    """
    if pts_xy.shape[0] == 0:
        return None, None

    ys = pts_xy[:, 1]
    if which == "top":
        region = pts_xy[ys <= np.percentile(ys, pct)]
    else:
        region = pts_xy[ys >= np.percentile(ys, 100.0 - pct)]

    if region.shape[0] < 2:
        m = pts_xy.mean(axis=0)
        return (m + np.array([-20, 0], np.float32)), (m + np.array([20, 0], np.float32))

    x_min = float(np.min(region[:, 0]))
    x_max = float(np.max(region[:, 0]))
    y_mean = float(np.mean(region[:, 1]))
    p1 = np.array([x_min, y_mean], dtype=np.float32)
    p2 = np.array([x_max, y_mean], dtype=np.float32)
    return p1, p2


def angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def signed_angle_between_lines(
    a1: np.ndarray, a2: np.ndarray,
    b1: np.ndarray, b2: np.ndarray,
    side: Optional[str] = None
) -> Tuple[float, float]:
    """
    두 선분 (a1->a2)와 (b1->b2)의 각도 차이를 반환.
    - abs_diff: 절댓값 (0~180)
    - signed_diff: 부호 포함 (-180~180)
    Right(R) side는 부호 flip (ipynb 규칙)
    """
    angA = angle_deg(a1, a2)
    angB = angle_deg(b1, b2)
    diff = angB - angA

    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360

    if side == "R":
        diff *= -1

    return abs(diff), diff


def knee_angle_3pts_deg(p_hip: np.ndarray, p_knee: np.ndarray, p_ankle: np.ndarray) -> float:
    """
    hip-knee-ankle 3점 각도 (0..180). straight ~180
    """
    v1 = p_hip - p_knee
    v2 = p_ankle - p_knee
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    cosang = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


# ---------------------------
# Mask-only refinements (ipynb 기반)
# ---------------------------
def fit_circle_least_squares(xy: np.ndarray) -> Tuple[float, float, float]:
    x = xy[:, 0]
    y = xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(max(float(c + cx**2 + cy**2), 1e-8))
    return float(cx), float(cy), float(r)


def get_femoral_head_points(
    pts: np.ndarray, side: str = "L", top_pct: float = 30.0, lr_frac: float = 0.50
) -> np.ndarray:
    """
    femur 상단(top_pct) 중 좌/우 lr_frac 구간만 선택하여 femoral head 후보 pts 추출
    """
    if pts.shape[0] == 0:
        return pts
    y_thr = np.percentile(pts[:, 1], top_pct)
    top_region = pts[pts[:, 1] <= y_thr]
    if top_region.shape[0] < 10:
        return top_region

    x_min, x_max = float(np.min(top_region[:, 0])), float(np.max(top_region[:, 0]))
    x_range = x_max - x_min + 1e-8

    if side == "L":
        x_cut = x_min + lr_frac * x_range
        region = top_region[top_region[:, 0] <= x_cut]
    else:
        x_cut = x_max - lr_frac * x_range
        region = top_region[top_region[:, 0] >= x_cut]

    return region if region.shape[0] >= 10 else top_region


def tibia_distal_plafond_midpoint(
    pts: np.ndarray, distal_pct: float = 20.0, band_frac: float = 0.30, extreme_frac: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    distal tibia(bottom band)에서 좌/우 extreme 평균을 뽑고, 그 중간을 ankle center(bot_mid)로 사용
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] == 0:
        z = np.zeros((2,), np.float32)
        return z, z, z

    y = pts[:, 1]
    y_thr = np.percentile(y, 100.0 - distal_pct)
    distal = pts[y >= y_thr]
    if distal.shape[0] < 20:
        bot_mid = distal.mean(axis=0) if distal.shape[0] else pts.mean(axis=0)
        return bot_mid.astype(np.float32), bot_mid.astype(np.float32), bot_mid.astype(np.float32)

    y_min = float(np.min(distal[:, 1]))
    y_max = float(np.max(distal[:, 1]))
    band_y = y_min + band_frac * (y_max - y_min)

    band = distal[distal[:, 1] <= band_y]
    if band.shape[0] < 10:
        band = distal

    xs = band[:, 0]
    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    x_range = x_max - x_min + 1e-8

    left_band = band[xs <= x_min + extreme_frac * x_range]
    right_band = band[xs >= x_max - extreme_frac * x_range]

    left_peak = left_band.mean(axis=0) if left_band.shape[0] else band[np.argmin(xs)]
    right_peak = right_band.mean(axis=0) if right_band.shape[0] else band[np.argmax(xs)]

    bot_mid = 0.5 * (left_peak + right_peak)
    return bot_mid.astype(np.float32), left_peak.astype(np.float32), right_peak.astype(np.float32)


# ---------------------------
# Feature extraction from masks
# ---------------------------
def extract_bone_features_from_masks(mask_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """
    mask_dict: {label: (H,W) uint8/0-1}
    return feats:
      feats[lbl]["pts"], ["top_mid"], ["bot_mid"]
      + tibia plafond peaks, femur head circle center(optional)
      + pelvis_top_p1/p2(optional)
    """
    feats: Dict[str, Dict] = {}

    for lbl, m in mask_dict.items():
        pts = mask_to_points(m)
        if pts.shape[0] < 20:
            continue

        top_mid = endpoint_midpoint_by_percentile(pts, which="top", pct=20.0)
        bot_mid = endpoint_midpoint_by_percentile(pts, which="bottom", pct=20.0)

        d = {"pts": pts, "top_mid": top_mid, "bot_mid": bot_mid}

        if lbl in ["Tibia_L", "Tibia_R"]:
            bot_mid2, left_peak, right_peak = tibia_distal_plafond_midpoint(
                pts, distal_pct=20.0, band_frac=0.30, extreme_frac=0.05
            )
            d["bot_mid"] = bot_mid2
            d["tibia_plafond_left"] = left_peak
            d["tibia_plafond_right"] = right_peak

        if lbl in ["Femur_L", "Femur_R"]:
            side = "L" if lbl.endswith("_L") else "R"
            head_pts = get_femoral_head_points(pts, side=side, top_pct=30.0, lr_frac=0.50)
            if head_pts.shape[0] >= 15:
                cx, cy, _r = fit_circle_least_squares(head_pts)
                d["femur_head_center"] = np.array([cx, cy], dtype=np.float32)
                # hip point를 femoral head center로 치환 (ipynb 흐름)
                d["top_mid"] = np.array([cx, cy], dtype=np.float32)

        feats[lbl] = d

    # Pelvis: mask 전체에서 좌/우 extreme 연결
    if "Pelvis" in feats and "Pelvis" in mask_dict:
        p1, p2 = pelvis_lr_endpoints_from_mask(mask_dict["Pelvis"])
        feats["Pelvis"]["pelvis_top_p1"] = p1
        feats["Pelvis"]["pelvis_top_p2"] = p2

    return feats


# ---------------------------
# Metric computations (degree)
# ---------------------------
def compute_alignment_metrics_from_feats(feats: Dict[str, Dict]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    # Pelvic tilt
    if "Pelvis" in feats:
        p1 = feats["Pelvis"].get("pelvis_top_p1", None)
        p2 = feats["Pelvis"].get("pelvis_top_p2", None)
        metrics["PelvicTilt_deg"] = angle_deg(p1, p2) if (p1 is not None and p2 is not None) else np.nan
    else:
        metrics["PelvicTilt_deg"] = np.nan

    def side_metrics(side: str) -> Dict[str, float]:
        fem_lbl = f"Femur_{side}"
        tib_lbl = f"Tibia_{side}"
        if fem_lbl not in feats or tib_lbl not in feats:
            return {"HKA_deg": np.nan, "mLDFA_deg": np.nan, "MPTA_deg": np.nan}

        fem_top = feats[fem_lbl].get("top_mid", None)
        fem_bot = feats[fem_lbl].get("bot_mid", None)
        tib_top = feats[tib_lbl].get("top_mid", None)
        tib_bot = feats[tib_lbl].get("bot_mid", None)
        if any(v is None for v in [fem_top, fem_bot, tib_top, tib_bot]):
            return {"HKA_deg": np.nan, "mLDFA_deg": np.nan, "MPTA_deg": np.nan}

        hip = fem_top
        knee = 0.5 * (fem_bot + tib_top)
        ankle = tib_bot
        HKA_value = knee_angle_3pts_deg(hip, knee, ankle)

        # mLDFA: femur axis vs distal femur joint line (bottom region)
        p1, p2 = joint_line_from_region(feats[fem_lbl]["pts"], which="bottom", pct=15.0)
        mLDFA_abs, _ = (
            signed_angle_between_lines(fem_top, fem_bot, p1, p2, side=side)
            if (p1 is not None and p2 is not None)
            else (np.nan, np.nan)
        )

        # MPTA: tibia axis vs proximal tibia joint line (top region)
        q1, q2 = joint_line_from_region(feats[tib_lbl]["pts"], which="top", pct=15.0)
        MPTA_abs, _ = (
            signed_angle_between_lines(tib_top, tib_bot, q1, q2, side=side)
            if (q1 is not None and q2 is not None)
            else (np.nan, np.nan)
        )

        return {"HKA_deg": HKA_value, "mLDFA_deg": mLDFA_abs, "MPTA_deg": MPTA_abs}

    L = side_metrics("L")
    R = side_metrics("R")

    metrics["HKA_L_deg"] = L["HKA_deg"]
    metrics["HKA_R_deg"] = R["HKA_deg"]
    metrics["mLDFA_L_deg"] = L["mLDFA_deg"]
    metrics["mLDFA_R_deg"] = R["mLDFA_deg"]
    metrics["MPTA_L_deg"] = L["MPTA_deg"]
    metrics["MPTA_R_deg"] = R["MPTA_deg"]

    return metrics


# ---------------------------
# Severity grading (LLD excluded)
# ---------------------------
def grade_by_deviation(dev, mild=3, moderate=5, severe=10):
    if np.isnan(dev):
        return "NA"
    if dev < mild:
        return "Normal"
    elif dev < moderate:
        return "Mild"
    elif dev < severe:
        return "Moderate"
    else:
        return "Severe"


def grade_HKA(HKA_value):
    if np.isnan(HKA_value):
        return "NA", np.nan
    dev = abs(HKA_value - 180.0)
    sev = grade_by_deviation(dev, mild=3, moderate=5, severe=10)
    return sev, dev


def grade_joint_angle(angle_value, normal_center=87, normal_band=3):
    if np.isnan(angle_value):
        return "NA", np.nan
    dev = abs(angle_value - normal_center)
    sev = grade_by_deviation(dev, mild=normal_band, moderate=5, severe=10)
    return sev, dev


def decide_deformity_source(mLDFA_dev, MPTA_dev):
    if np.isnan(mLDFA_dev) and np.isnan(MPTA_dev):
        return "Unknown"

    fem_bad = (not np.isnan(mLDFA_dev)) and (mLDFA_dev >= 3)
    tib_bad = (not np.isnan(MPTA_dev)) and (MPTA_dev >= 3)

    if fem_bad and tib_bad:
        if mLDFA_dev > MPTA_dev + 1:
            return "Femur-dominant"
        elif MPTA_dev > mLDFA_dev + 1:
            return "Tibia-dominant"
        else:
            return "Both"
    elif fem_bad:
        return "Femur"
    elif tib_bad:
        return "Tibia"
    else:
        return "Well-aligned"


def add_severity_labels(metrics_row: Dict) -> Dict:
    """
    Input keys expected:
      HKA_L_deg, HKA_R_deg, mLDFA_L_deg, mLDFA_R_deg, MPTA_L_deg, MPTA_R_deg
    Output: deviations + severity labels + deformity sources + overall severity.
    """
    out = {}

    sev_hka_L, dev_hka_L = grade_HKA(metrics_row.get("HKA_L_deg", np.nan))
    sev_hka_R, dev_hka_R = grade_HKA(metrics_row.get("HKA_R_deg", np.nan))

    out["HKA_L_dev_deg"] = dev_hka_L
    out["HKA_R_dev_deg"] = dev_hka_R
    out["HKA_L_severity"] = sev_hka_L
    out["HKA_R_severity"] = sev_hka_R

    sev_mldfa_L, dev_mldfa_L = grade_joint_angle(metrics_row.get("mLDFA_L_deg", np.nan))
    sev_mldfa_R, dev_mldfa_R = grade_joint_angle(metrics_row.get("mLDFA_R_deg", np.nan))
    sev_mpta_L,  dev_mpta_L  = grade_joint_angle(metrics_row.get("MPTA_L_deg", np.nan))
    sev_mpta_R,  dev_mpta_R  = grade_joint_angle(metrics_row.get("MPTA_R_deg", np.nan))

    out["mLDFA_L_dev_deg"] = dev_mldfa_L
    out["mLDFA_R_dev_deg"] = dev_mldfa_R
    out["MPTA_L_dev_deg"]  = dev_mpta_L
    out["MPTA_R_dev_deg"]  = dev_mpta_R

    out["mLDFA_L_severity"] = sev_mldfa_L
    out["mLDFA_R_severity"] = sev_mldfa_R
    out["MPTA_L_severity"]  = sev_mpta_L
    out["MPTA_R_severity"]  = sev_mpta_R

    out["DeformitySource_L"] = decide_deformity_source(dev_mldfa_L, dev_mpta_L)
    out["DeformitySource_R"] = decide_deformity_source(dev_mldfa_R, dev_mpta_R)

    def worst_severity(*labels):
        order = {"NA": -1, "Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        return max(labels, key=lambda x: order.get(x, -1))

    out["OverallAlignSeverity_L"] = worst_severity(sev_hka_L, sev_mldfa_L, sev_mpta_L)
    out["OverallAlignSeverity_R"] = worst_severity(sev_hka_R, sev_mldfa_R, sev_mpta_R)

    out["SeverityDriver"] = (
        "Angle (Alignment)"
        if (out["OverallAlignSeverity_L"] in ["Moderate", "Severe"]
            or out["OverallAlignSeverity_R"] in ["Moderate", "Severe"])
        else "Normal/Mild"
    )

    return out


def _largest_contour_from_mask(mask01: np.ndarray) -> Optional[np.ndarray]:
    """
    mask01: (H,W) uint8 {0,1}
    return: contour points (N,2) in (x,y) float32 for the largest contour
    """
    m = (mask01 > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    # (N,1,2) -> (N,2)
    pts = c.reshape(-1, 2).astype(np.float32)
    return pts

def pelvis_top_line_from_contour(mask01: np.ndarray, top_band_pct: float = 25.0, make_horizontal: bool = True
                                 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Pelvis의 상단 경계(contour)에서 top band만 취해 좌우 끝점을 반환.
    - top_band_pct: y 기준 상단 몇 %를 top band로 볼지
    - make_horizontal: True면 두 끝점의 y를 평균으로 맞춰 수평선으로 만듦
    return: (p1, p2) each (2,) float32 in (x,y)
    """
    pts = _largest_contour_from_mask(mask01)
    if pts is None or pts.shape[0] < 10:
        return None, None

    ys = pts[:, 1]
    y_thr = np.percentile(ys, top_band_pct)  # 상단 band
    band = pts[ys <= y_thr]
    if band.shape[0] < 2:
        band = pts

    left = band[np.argmin(band[:, 0])]
    right = band[np.argmax(band[:, 0])]

    if make_horizontal:
        y0 = float(np.mean([left[1], right[1]]))
        left = np.array([left[0], y0], dtype=np.float32)
        right = np.array([right[0], y0], dtype=np.float32)

    return left.astype(np.float32), right.astype(np.float32)

def pelvis_lr_endpoints_from_mask(mask01: np.ndarray):
    """
    Pelvis mask에서 가장 왼쪽 / 오른쪽 점 반환
    return: (left_xy, right_xy) or (None, None)
    """
    pts = mask_to_points(mask01)
    if pts.shape[0] < 2:
        return None, None

    left = pts[np.argmin(pts[:, 0])]
    right = pts[np.argmax(pts[:, 0])]
    return left.astype(np.float32), right.astype(np.float32)
