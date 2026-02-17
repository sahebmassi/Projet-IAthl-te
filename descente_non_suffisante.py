import argparse
import os
import time
from typing import List, Tuple, Optional
from collections import deque
from statistics import median

import cv2
from ultralytics import YOLO


# ============================================================
# Paramètres (à tuner) pour détection de "descente non suffisante"
# ============================================================
# Durée minimale d'une descente valide (persistance)
MIN_DESC_FRAMES = 6  # en frames (sera adapté via FPS)

# Seuil profondeur minimale de la descente, en ratio de la largeur de hanches
MIN_DESCENT_RATIO_OF_HIPWIDTH = 0.10  # 10% hipWidth

# EPS : seuil "vitesse" par frame pour compter un mouvement vers le bas, en % hipWidth/frame
EPS_VEL_RATIO_OF_HIPWIDTH = 0.008

# Lissage (médiane glissante)
SMOOTH_WIN = 5

# Détection de phase : nb de frames consécutives pour valider descente/remontée
N_DOWN_FRAMES = 4
N_UP_FRAMES = 4


# ============================================================
# Paramètres fin de remontée (réutilisés pour stabiliser la fin)
# ============================================================
LOCK_ANGLE_DEG = 173.0
LOCK_HOLD_SEC = 0.12
TOP_BAND_RATIO_OF_HIPWIDTH = 0.035
TOP_HOLD_SEC = 0.12
END_VEL_RATIO_OF_HIPWIDTH = 0.006
MIN_ASCENT_BEFORE_END_SEC = 0.25


# Squelette YOLO COCO17
YOLO_COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

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
    if not keypoints or len(keypoints) < 17:
        return

    for idx, (x, y) in enumerate(keypoints):
        if x is None or y is None or (x == 0 and y == 0):
            continue
        cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(
            image, str(idx), (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
        )

    for a, b in YOLO_COCO17_SKELETON:
        xa, ya = keypoints[a]
        xb, yb = keypoints[b]
        if (xa == 0 and ya == 0) or (xb == 0 and yb == 0):
            continue
        cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)


def pick_best_person(result) -> Optional[List[Tuple[float, float]]]:
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

    norm_ba = (ba[0] * ba[0] + ba[1] * ba[1]) ** 0.5
    norm_bc = (bc[0] * bc[0] + bc[1] * bc[1]) ** 0.5
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cosang = (ba[0] * bc[0] + ba[1] * bc[1]) / (norm_ba * norm_bc)
    cosang = clamp(cosang, -1.0, 1.0)
    return float((180.0 / 3.141592653589793) * __import__("math").acos(cosang))


def get_midhip_and_hipwidth(kps: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    xL, yL = kps[LEFT_HIP]
    xR, yR = kps[RIGHT_HIP]
    if (xL == 0 and yL == 0) or (xR == 0 and yR == 0):
        return None, None

    mid_y = 0.5 * (yL + yR)

    dx = xL - xR
    dy = yL - yR
    hip_width = (dx * dx + dy * dy) ** 0.5
    if hip_width < 1.0:
        hip_width = None

    return float(mid_y), float(hip_width) if hip_width is not None else None


def get_avg_knee_angle(kps: List[Tuple[float, float]]) -> Optional[float]:
    left = angle_deg(kps[LEFT_HIP], kps[LEFT_KNEE], kps[LEFT_ANKLE])
    right = angle_deg(kps[RIGHT_HIP], kps[RIGHT_KNEE], kps[RIGHT_ANKLE])

    vals = [v for v in (left, right) if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Chemin vidéo (optionnel). Si absent: boîte de dialogue.")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Modèle YOLO pose (.pt/.onnx).")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil confiance YOLO")
    parser.add_argument("--device", default=None, help="cpu / cuda / mps (optionnel)")
    args = parser.parse_args()

    video_path = args.video or select_video_with_dialog()
    if not video_path:
        print("Aucune vidéo sélectionnée. Exit.")
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
    print(f"[INFO] FPS vidéo = {fps_video:.2f}")

    MIN_DESC_FRAMES_FR = max(2, int(round(MIN_DESC_FRAMES)))
    LOCK_HOLD_FRAMES = max(2, int(round(LOCK_HOLD_SEC * fps_video)))
    TOP_HOLD_FRAMES = max(2, int(round(TOP_HOLD_SEC * fps_video)))
    MIN_ASCENT_BEFORE_END_FRAMES = max(0, int(round(MIN_ASCENT_BEFORE_END_SEC * fps_video)))

    print(f"[INFO] Detection params: min_desc_frames={MIN_DESC_FRAMES_FR} | min_descent_ratio={MIN_DESCENT_RATIO_OF_HIPWIDTH*100:.1f}% hipWidth")

    win = "Squat - descente non suffisante"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    hip_y_buf = deque(maxlen=SMOOTH_WIN)
    hipw_buf = deque(maxlen=SMOOTH_WIN)
    knee_buf = deque(maxlen=SMOOTH_WIN)
    baseline_buf = deque(maxlen=int(max(10, round(1.0 * fps_video))))

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

    baseline_top_y = None

    descent_insufficient = False
    descent_amp_px = None

    lock_streak = 0
    top_streak = 0

    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            results = model.predict(source=frame, conf=args.conf, verbose=False, device=args.device)
            res0 = results[0] if results else None

            kps = pick_best_person(res0)
            if kps is not None:
                draw_skeleton(frame, kps)

                mid_y, hipw = get_midhip_and_hipwidth(kps)
                if mid_y is not None:
                    hip_y_buf.append(mid_y)
                    baseline_buf.append(mid_y)
                    if hipw is not None:
                        hipw_buf.append(hipw)

                knee = get_avg_knee_angle(kps)
                if knee is not None:
                    knee_buf.append(knee)

            if len(hip_y_buf) >= max(3, SMOOTH_WIN // 2):
                smooth_y = float(median(hip_y_buf))
                hipw_med = float(median(hipw_buf)) if len(hipw_buf) > 0 else 200.0

                eps_px = EPS_VEL_RATIO_OF_HIPWIDTH * hipw_med
                end_vel_eps = END_VEL_RATIO_OF_HIPWIDTH * hipw_med
                top_band_px = TOP_BAND_RATIO_OF_HIPWIDTH * hipw_med

                smooth_knee = float(median(knee_buf)) if len(knee_buf) > 0 else None

                if prev_smooth_y is not None:
                    vel = smooth_y - prev_smooth_y  # >0: descend / <0: remonte

                    if vel > eps_px:
                        down_streak += 1
                    else:
                        down_streak = 0

                    if vel < -eps_px:
                        up_streak += 1
                    else:
                        up_streak = 0

                    if state == "idle":
                        if down_streak >= N_DOWN_FRAMES:
                            state = "descending"
                            start_desc_frame = frame_idx - N_DOWN_FRAMES + 1
                            if baseline_buf:
                                baseline_top_y = float(median(baseline_buf))
                            else:
                                baseline_top_y = prev_smooth_y

                            max_smooth_y = smooth_y
                            max_smooth_y_frame = frame_idx
                            print(f"[INFO] Descente détectée à frame={start_desc_frame} (t={start_desc_frame/fps_video:.2f}s)")

                    elif state == "descending":
                        if max_smooth_y is None or smooth_y > max_smooth_y:
                            max_smooth_y = smooth_y
                            max_smooth_y_frame = frame_idx

                        # autoriser remontée seulement après un nombre minimal de frames
                        if start_desc_frame is not None and (frame_idx - start_desc_frame) >= MIN_DESC_FRAMES_FR:
                            if up_streak >= N_UP_FRAMES:
                                state = "ascending"
                                start_up_frame = frame_idx - N_UP_FRAMES + 1

                                bottom_info = ""
                                if max_smooth_y_frame is not None:
                                    bottom_info = f" | bottom≈frame={max_smooth_y_frame} (t={max_smooth_y_frame/fps_video:.2f}s)"
                                print(f"[OK] Début REMONTÉE à frame={start_up_frame} (t={start_up_frame/fps_video:.2f}s){bottom_info}")

                                # Évaluer amplitude de la descente
                                if baseline_top_y is not None and max_smooth_y is not None:
                                    amp = max_smooth_y - baseline_top_y
                                    descent_amp_px = amp
                                    if amp < (MIN_DESCENT_RATIO_OF_HIPWIDTH * hipw_med):
                                        descent_insufficient = True
                                        print(
                                            f"[FAULT] Descente non suffisante: amp={amp:.1f}px "
                                            f"(< {MIN_DESCENT_RATIO_OF_HIPWIDTH*100:.1f}% hipWidth ~ {MIN_DESCENT_RATIO_OF_HIPWIDTH*hipw_med:.1f}px)"
                                        )

                                # reset end tracking
                                lock_streak = 0
                                top_streak = 0

                    elif state == "ascending":
                        # Detection FIN de remontée (stabilité)
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
                                print(f"[DONE] Fin remontée détectée ({reason}) à frame={end_frame} (t={end_frame/fps_video:.2f}s)")

                prev_smooth_y = smooth_y

                # Overlay
                cv2.putText(frame, f"state: {state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if start_desc_frame is not None:
                    cv2.putText(frame, f"DESC START: frame {start_desc_frame}  t={start_desc_frame/fps_video:.2f}s",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if start_up_frame is not None:
                    cv2.putText(frame, f"ASCENT START: frame {start_up_frame}  t={start_up_frame/fps_video:.2f}s",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if end_frame is not None:
                    cv2.putText(frame, f"ASCENT END: frame {end_frame}  t={end_frame/fps_video:.2f}s",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if smooth_knee is not None:
                    cv2.putText(frame, f"knee angle ~ {smooth_knee:.1f} deg", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                if descent_insufficient:
                    cv2.putText(frame, "FAULT: DESCENTE NON SUFFISANTE", (10, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            cv2.putText(frame, "Space: pause | R: reset | Q/Esc: quit", (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

        if key == ord("r"):
            state = "idle"
            hip_y_buf.clear()
            hipw_buf.clear()
            knee_buf.clear()
            baseline_buf.clear()

            prev_smooth_y = None
            down_streak = 0
            up_streak = 0

            start_desc_frame = None
            start_up_frame = None
            end_frame = None

            max_smooth_y = None
            max_smooth_y_frame = None

            descent_insufficient = False
            descent_amp_px = None

            lock_streak = 0
            top_streak = 0

            print("[INFO] Reset état/mesures.")

    cap.release()
    cv2.destroyAllWindows()

    # Résultat final
    print("\n========== RÉSULTAT ==========")
    if start_desc_frame is None:
        print("Descente non détectée -> pas d'analyse de profondeur.")
        return

    print(f"Début descente: frame={start_desc_frame}  t={start_desc_frame/fps_video:.2f}s")

    if max_smooth_y_frame is not None:
        print(f"Bottom approximatif: frame={max_smooth_y_frame} t={max_smooth_y_frame/fps_video:.2f}s")

    if descent_amp_px is not None:
        print(f"Amplitude descente (px): ~{descent_amp_px:.1f}")
    else:
        print("Amplitude descente: non mesurée")

    if descent_insufficient:
        print("FAUTE: descente insuffisante détectée.")
        print(f"  - amplitude ~ {descent_amp_px:.1f}px (< {MIN_DESCENT_RATIO_OF_HIPWIDTH*100:.1f}% hipWidth)")
    else:
        print("OK: descente suffisante détectée (au-dessus du seuil).")


if __name__ == "__main__":
    main()
