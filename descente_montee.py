import argparse
import os
import math
from typing import List, Tuple, Optional
from collections import deque
from statistics import median

import cv2
import numpy as np
from ultralytics import YOLO

# ✅ Option 2 : texte Unicode (accents) via Pillow + police TTF
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 1) PARAMÈTRES À TUNER (DIP pendant la montée)
# ============================================================
K_MIN_DIP_SEC = 0.06
X_IGNORE_AFTER_ASCENT_SEC = 0.05
A_MIN_DIP_RATIO_OF_HIPWIDTH = 0.015
EPS_VEL_RATIO_OF_HIPWIDTH = 0.008

SMOOTH_WIN = 5
N_DOWN_FRAMES = 4
N_UP_FRAMES = 4
MIN_DESC_FRAMES = 10


# ============================================================
# 2) PARAMÈTRES À TUNER (FIN de la remontée)
# ============================================================
LOCK_ANGLE_DEG = 173.0
LOCK_HOLD_SEC = 0.12

TOP_BAND_RATIO_OF_HIPWIDTH = 0.035
TOP_HOLD_SEC = 0.12

END_VEL_RATIO_OF_HIPWIDTH = 0.006
MIN_ASCENT_BEFORE_END_SEC = 0.25


# ============================================================
# 3) AFFICHAGE (panneau info DANS LA MÊME FENÊTRE)
# ============================================================
PANEL_W = 560
PANEL_BG_BGR = (28, 28, 28)
PANEL_FG_RGB = (235, 235, 235)
PANEL_LINE_RGB = (90, 90, 90)
PANEL_TITLE_RGB = (255, 255, 255)


# Squelette COCO 17 (comme ton zip)
YOLO_COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Indices YOLO (COCO17)
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


def select_video_with_dialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise RuntimeError(
            "Tkinter indisponible. Installe python3-tk (Linux/WSL) ou passe --video."
        ) from e

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Sélectionner une vidéo",
        filetypes=[
            ("Vidéos", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm"),
            ("Tous les fichiers", "*.*"),
        ],
    )
    root.destroy()
    return path


def draw_skeleton(image, keypoints: List[Tuple[float, float]]) -> None:
    """Dessine squelette + indices (digits => pas de souci d'accents ici)."""
    if not keypoints or len(keypoints) < 17:
        return

    for idx, (x, y) in enumerate(keypoints):
        if x is None or y is None or (x == 0 and y == 0):
            continue
        cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(image, str(idx), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    for a, b in YOLO_COCO17_SKELETON:
        xa, ya = keypoints[a]
        xb, yb = keypoints[b]
        if (xa == 0 and ya == 0) or (xb == 0 and yb == 0):
            continue
        cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)


def pick_best_person(result) -> Optional[List[Tuple[float, float]]]:
    """Choisit la personne principale (aire bbox max)."""
    if result is None or result.keypoints is None or result.keypoints.xy is None:
        return None
    kps_xy = result.keypoints.xy
    if len(kps_xy) == 0:
        return None

    if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
        person = kps_xy[0].cpu().numpy().tolist()
        return [(float(x), float(y)) for x, y in person]

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(areas.argmax())

    person = kps_xy[best_idx].cpu().numpy().tolist()
    return [(float(x), float(y)) for x, y in person]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Optional[float]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    if (ax == 0 and ay == 0) or (bx == 0 and by == 0) or (cx == 0 and cy == 0):
        return None

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    norm_ba = math.hypot(ba[0], ba[1])
    norm_bc = math.hypot(bc[0], bc[1])
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cosang = (ba[0] * bc[0] + ba[1] * bc[1]) / (norm_ba * norm_bc)
    cosang = clamp(cosang, -1.0, 1.0)
    return float(math.degrees(math.acos(cosang)))


def get_midhip_and_hipwidth(kps: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    xL, yL = kps[LEFT_HIP]
    xR, yR = kps[RIGHT_HIP]
    if (xL == 0 and yL == 0) or (xR == 0 and yR == 0):
        return None, None

    mid_y = 0.5 * (yL + yR)
    hipw = math.hypot(xL - xR, yL - yR)
    if hipw < 1.0:
        hipw = None
    return float(mid_y), float(hipw) if hipw is not None else None


def get_avg_knee_angle(kps: List[Tuple[float, float]]) -> Optional[float]:
    left = angle_deg(kps[LEFT_HIP], kps[LEFT_KNEE], kps[LEFT_ANKLE])
    right = angle_deg(kps[RIGHT_HIP], kps[RIGHT_KNEE], kps[RIGHT_ANKLE])
    vals = [v for v in (left, right) if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def find_ttf_font() -> Optional[str]:
    """Essaie de trouver une police TTF connue (Linux/WSL/Mac/Windows)."""
    candidates = [
        # Linux / WSL (souvent dispo)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # Mac
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Windows (si jamais)
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def build_fonts(height: int):
    """Crée deux polices (title + body) adaptées à la hauteur."""
    font_path = find_ttf_font()
    # tailles ajustées selon H
    if height >= 900:
        body_size, title_size = 22, 28
    elif height >= 720:
        body_size, title_size = 20, 26
    else:
        body_size, title_size = 18, 24

    if font_path:
        body = ImageFont.truetype(font_path, body_size)
        title = ImageFont.truetype(font_path, title_size)
        return title, body, font_path
    else:
        # fallback (peut être moins bon pour accents, mais on tente)
        body = ImageFont.load_default()
        title = ImageFont.load_default()
        return title, body, None


def make_panel_unicode(height: int, lines: List[str], title: str, fonts_cache: dict) -> np.ndarray:
    """
    Panneau BGR (numpy) rendu via PIL (Unicode OK).
    """
    # cache fonts par hauteur
    if height not in fonts_cache:
        title_font, body_font, fp = build_fonts(height)
        fonts_cache[height] = (title_font, body_font, fp)

    title_font, body_font, font_path = fonts_cache[height]

    # BGR -> RGB pour PIL
    panel_bgr = np.zeros((height, PANEL_W, 3), dtype=np.uint8)
    panel_bgr[:] = PANEL_BG_BGR
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(panel_rgb)
    draw = ImageDraw.Draw(img)

    x = 16
    y = 12
    draw.text((x, y), title, font=title_font, fill=PANEL_TITLE_RGB)
    y += 38

    # ligne séparatrice
    draw.line((x, y, PANEL_W - 16, y), fill=PANEL_LINE_RGB, width=2)
    y += 16

    # petit warning si pas de font ttf
    if font_path is None:
        draw.text((x, y), "⚠ Police TTF introuvable : accents peuvent bugger.", font=body_font, fill=(255, 200, 120))
        y += 28

    # body lines
    line_gap = 8
    for s in lines:
        if y > height - 24:
            break
        draw.text((x, y), s, font=body_font, fill=PANEL_FG_RGB)
        # mesure hauteur texte
        bbox = draw.textbbox((x, y), s, font=body_font)
        text_h = bbox[3] - bbox[1]
        y += text_h + line_gap

    # PIL RGB -> numpy BGR
    out_rgb = np.array(img)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Chemin vidéo (optionnel). Si absent: boîte de dialogue.")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Modèle YOLO pose (.pt/.onnx).")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil confiance YOLO")
    parser.add_argument("--device", default=None, help="cpu / cuda / mps (optionnel)")
    args = parser.parse_args()

    video_path = args.video or select_video_with_dialog()
    if not video_path:
        return
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Vidéo introuvable: {video_path}")

    model = YOLO(args.model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la vidéo.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_video = float(cap.get(cv2.CAP_PROP_FPS))
    if fps_video <= 1e-6:
        fps_video = 30.0

    # Conversions temps -> frames
    K_MIN_DIP_FRAMES = max(2, int(round(K_MIN_DIP_SEC * fps_video)))
    X_IGNORE_FRAMES = max(0, int(round(X_IGNORE_AFTER_ASCENT_SEC * fps_video)))
    LOCK_HOLD_FRAMES = max(2, int(round(LOCK_HOLD_SEC * fps_video)))
    TOP_HOLD_FRAMES = max(2, int(round(TOP_HOLD_SEC * fps_video)))
    MIN_ASCENT_BEFORE_END_FRAMES = max(0, int(round(MIN_ASCENT_BEFORE_END_SEC * fps_video)))

    # Une seule fenêtre (mosaïque)
    win = "Squat (Vidéo + Info Unicode)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W + PANEL_W, H)

    # Buffers lissage
    hip_y_buf = deque(maxlen=SMOOTH_WIN)
    hipw_buf = deque(maxlen=SMOOTH_WIN)
    knee_buf = deque(maxlen=SMOOTH_WIN)

    baseline_buf = deque(maxlen=int(max(10, round(1.0 * fps_video))))
    baseline_top_y = None

    prev_smooth_y = None

    state = "idle"  # idle -> descending -> ascending -> done
    frame_idx = -1

    down_streak = 0
    up_streak = 0

    start_desc_frame = None
    start_up_frame = None
    end_frame = None

    max_smooth_y = None
    max_smooth_y_frame = None

    # Dip
    dip_detected = False
    dip_start_frame = None
    dip_end_frame = None
    dip_amp_px = None

    dip_streak = 0
    dip_candidate_start = None
    dip_base_y = None
    dip_peak_y = None

    # End
    lock_streak = 0
    top_streak = 0

    # “logs” affichés dans le panneau
    events = deque(maxlen=10)

    def log_event(msg: str):
        t = (max(frame_idx, 0) / fps_video)
        events.appendleft(f"[{t:5.2f}s] {msg}")

    paused = False
    fonts_cache = {}

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            frame_clean = frame.copy()

            results = model.predict(source=frame_clean, conf=args.conf, verbose=False, device=args.device)
            res0 = results[0] if results else None

            kps = pick_best_person(res0)
            if kps is not None:
                draw_skeleton(frame_clean, kps)

                mid_y, hipw = get_midhip_and_hipwidth(kps)
                if mid_y is not None:
                    hip_y_buf.append(mid_y)
                    baseline_buf.append(mid_y)
                    if hipw is not None:
                        hipw_buf.append(hipw)

                knee = get_avg_knee_angle(kps)
                if knee is not None:
                    knee_buf.append(knee)

            smooth_y = None
            hipw_med = None
            smooth_knee = None
            vel = None

            if len(hip_y_buf) >= max(3, SMOOTH_WIN // 2):
                smooth_y = float(median(hip_y_buf))
                hipw_med = float(median(hipw_buf)) if len(hipw_buf) > 0 else 200.0
                smooth_knee = float(median(knee_buf)) if len(knee_buf) > 0 else None

                eps_px = EPS_VEL_RATIO_OF_HIPWIDTH * hipw_med
                A_px = A_MIN_DIP_RATIO_OF_HIPWIDTH * hipw_med
                end_vel_eps = END_VEL_RATIO_OF_HIPWIDTH * hipw_med
                top_band_px = TOP_BAND_RATIO_OF_HIPWIDTH * hipw_med

                if prev_smooth_y is not None:
                    vel = smooth_y - prev_smooth_y  # >0 descend, <0 remonte

                    down_streak = down_streak + 1 if vel > eps_px else 0
                    up_streak = up_streak + 1 if vel < -eps_px else 0

                    if state == "idle":
                        if down_streak >= N_DOWN_FRAMES:
                            state = "descending"
                            start_desc_frame = frame_idx - N_DOWN_FRAMES + 1
                            baseline_top_y = float(median(baseline_buf)) if baseline_buf else prev_smooth_y
                            max_smooth_y = smooth_y
                            max_smooth_y_frame = frame_idx
                            log_event(f"Descente détectée @ frame {start_desc_frame}")

                    elif state == "descending":
                        if max_smooth_y is None or smooth_y > max_smooth_y:
                            max_smooth_y = smooth_y
                            max_smooth_y_frame = frame_idx

                        if start_desc_frame is not None and (frame_idx - start_desc_frame) >= MIN_DESC_FRAMES:
                            if up_streak >= N_UP_FRAMES:
                                state = "ascending"
                                start_up_frame = frame_idx - N_UP_FRAMES + 1
                                log_event(f"Début REMONTÉE @ frame {start_up_frame} (bottom~{max_smooth_y_frame})")

                                dip_streak = 0
                                dip_candidate_start = None
                                dip_base_y = None
                                dip_peak_y = None

                                lock_streak = 0
                                top_streak = 0

                    elif state == "ascending":
                        # FIN remontée
                        if start_up_frame is not None and (frame_idx - start_up_frame) >= MIN_ASCENT_BEFORE_END_FRAMES:
                            stable = abs(vel) < end_vel_eps

                            if smooth_knee is not None and smooth_knee >= LOCK_ANGLE_DEG and stable:
                                lock_streak += 1
                            else:
                                lock_streak = 0

                            if baseline_top_y is not None and abs(smooth_y - baseline_top_y) <= top_band_px and stable:
                                top_streak += 1
                            else:
                                top_streak = 0

                            if lock_streak >= LOCK_HOLD_FRAMES or top_streak >= TOP_HOLD_FRAMES:
                                state = "done"
                                end_frame = frame_idx
                                reason = "LOCK" if lock_streak >= LOCK_HOLD_FRAMES else "TOP"
                                log_event(f"FIN remontée ({reason}) @ frame {end_frame}")

                        # DIP (si pas done)
                        if state == "ascending" and start_up_frame is not None and not dip_detected:
                            if (frame_idx - start_up_frame) >= X_IGNORE_FRAMES:
                                if vel > eps_px:
                                    if dip_candidate_start is None:
                                        dip_candidate_start = frame_idx
                                        dip_base_y = prev_smooth_y
                                        dip_peak_y = smooth_y
                                    dip_streak += 1
                                    dip_peak_y = max(dip_peak_y, smooth_y)
                                else:
                                    if dip_candidate_start is not None:
                                        amp = (dip_peak_y - dip_base_y) if (dip_peak_y is not None and dip_base_y is not None) else 0.0
                                        if dip_streak >= K_MIN_DIP_FRAMES and amp >= A_px:
                                            dip_detected = True
                                            dip_start_frame = dip_candidate_start
                                            dip_end_frame = frame_idx
                                            dip_amp_px = amp
                                            log_event(f"FAUTE : dip détecté @ frame {dip_start_frame} (amp~{amp:.1f}px)")

                                        dip_streak = 0
                                        dip_candidate_start = None
                                        dip_base_y = None
                                        dip_peak_y = None

                prev_smooth_y = smooth_y

            # ----- Panneau info unicode -----
            t_sec = frame_idx / fps_video
            hipw_txt = f"{hipw_med:.1f}px" if hipw_med is not None else "-"
            vel_txt = f"{vel:+.2f}px/frame" if vel is not None else "-"
            knee_txt = f"{smooth_knee:.1f}°" if smooth_knee is not None else "-"

            lines = [
                f"FPS vidéo : {fps_video:.2f}",
                f"Frame : {frame_idx}   Temps : {t_sec:.2f}s",
                f"État : {state}",
                "",
                f"Début remontée : {start_up_frame if start_up_frame is not None else '-'}",
                f"Fin remontée   : {end_frame if end_frame is not None else '-'}",
                f"Angle genou    : {knee_txt}",
                f"HipWidth       : {hipw_txt}",
                f"Vitesse (hipY) : {vel_txt}",
                "",
                "DIP (redescente pendant montée) :",
                f"  Détecté : {'OUI' if dip_detected else 'NON'}",
                f"  Début   : {dip_start_frame if dip_start_frame is not None else '-'}",
                f"  Amp     : {f'{dip_amp_px:.1f}px' if dip_amp_px is not None else '-'}",
                "",
                "Événements (logs) :",
            ]
            lines.extend(list(events))

            panel = make_panel_unicode(H, lines, title="TABLEAU DE BORD (accents OK)", fonts_cache=fonts_cache)

            combo = np.hstack([frame_clean, panel])
            cv2.imshow(win, combo)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            log_event("Pause activée" if paused else "Pause désactivée")

        if key == ord("r"):
            # reset + rewind
            state = "idle"
            frame_idx = -1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            hip_y_buf.clear()
            hipw_buf.clear()
            knee_buf.clear()
            baseline_buf.clear()
            baseline_top_y = None
            prev_smooth_y = None

            down_streak = 0
            up_streak = 0
            start_desc_frame = None
            start_up_frame = None
            end_frame = None
            max_smooth_y = None
            max_smooth_y_frame = None

            dip_detected = False
            dip_start_frame = None
            dip_end_frame = None
            dip_amp_px = None
            dip_streak = 0
            dip_candidate_start = None
            dip_base_y = None
            dip_peak_y = None

            lock_streak = 0
            top_streak = 0

            events.clear()
            log_event("RESET + retour au début")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
