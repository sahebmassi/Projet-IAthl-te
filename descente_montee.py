import argparse
import os
import time
from typing import List, Tuple, Optional
from collections import deque
from statistics import median

import cv2
from ultralytics import YOLO


# ------------------------------
# 1) PARAMÈTRES À TUNER ICI ✅
# ------------------------------
# Ces paramètres contrôlent la détection du "dip" (redescente pendant la remontée).
# Ils sont exprimés en SECONDES et en RATIOS normalisés (pas en pixels bruts),
# puis convertis en "frames" automatiquement via le FPS lu depuis la vidéo.

# K : durée minimale d’un dip (persistance). (ex: 0.12s ~ 4 frames à 30fps)
K_MIN_DIP_SEC = 0.06

# X : zone tampon après le début de la remontée (ignore les toutes premières frames)
# pour éviter de confondre une micro oscillation au bottom avec une vraie redescente.
X_IGNORE_AFTER_ASCENT_SEC = 0.05

# A : amplitude minimale du dip (tolérance anti-jitter)
# exprimée comme % de la largeur du bassin (distance hipL-hipR).
A_MIN_DIP_RATIO_OF_HIPWIDTH = 0.03  # 3% de la largeur bassin (à ajuster)

# EPS : seuil de "vitesse" (delta Y par frame) pour considérer un mouvement réel
# exprimé comme % de la largeur du bassin PAR FRAME.
EPS_VEL_RATIO_OF_HIPWIDTH = 0.008

# Lissage du signal (plus grand = plus stable mais plus de retard)
SMOOTH_WIN = 3  # fenêtre médiane (impair conseillé)

# Détection de phase (descente/remontée) : combien de frames consécutives
N_DOWN_FRAMES = 4
N_UP_FRAMES = 4

# On évite de détecter la remontée trop tôt (descente minimum)
MIN_DESC_FRAMES = 10
# ------------------------------


# Squelette identique à ton zip (COCO 17)
YOLO_COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

LEFT_HIP, RIGHT_HIP = 11, 12


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

    kps_xy = result.keypoints.xy  # (n_person, 17, 2)
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


def get_midhip_and_hipwidth(kps: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    (xL, yL) = kps[LEFT_HIP]
    (xR, yR) = kps[RIGHT_HIP]
    if (xL == 0 and yL == 0) or (xR == 0 and yR == 0):
        return None, None

    mid_y = 0.5 * (yL + yR)

    dx = xL - xR
    dy = yL - yR
    hip_width = (dx * dx + dy * dy) ** 0.5
    if hip_width < 1.0:
        hip_width = None

    return float(mid_y), float(hip_width) if hip_width is not None else None


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

    # ✅ FPS lu depuis la vidéo
    fps_video = float(cap.get(cv2.CAP_PROP_FPS))
    if fps_video <= 1e-6:
        fps_video = 30.0  # fallback si FPS non disponible
    print(f"[INFO] FPS vidéo = {fps_video:.2f}")

    # Conversion des paramètres temps -> frames (automatique)
    K_MIN_DIP_FRAMES = max(3, int(round(K_MIN_DIP_SEC * fps_video)))
    X_IGNORE_FRAMES = max(0, int(round(X_IGNORE_AFTER_ASCENT_SEC * fps_video)))

    print(f"[INFO] Paramètres (auto via FPS): K={K_MIN_DIP_FRAMES} frames, X={X_IGNORE_FRAMES} frames, "
          f"A={A_MIN_DIP_RATIO_OF_HIPWIDTH*100:.1f}% hipWidth")

    win = "Squat - ascent start + dip detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    hip_y_buf = deque(maxlen=SMOOTH_WIN)
    hipw_buf = deque(maxlen=SMOOTH_WIN)

    prev_smooth_y = None

    state = "idle"  # idle -> descending -> ascending
    frame_idx = -1

    down_streak = 0
    up_streak = 0

    start_desc_frame = None
    start_up_frame = None

    # bottom info (optionnel)
    max_smooth_y = None
    max_smooth_y_frame = None

    # ---- Dip detection state (pendant montée) ----
    dip_detected = False
    dip_start_frame = None
    dip_end_frame = None
    dip_amp_px = None

    dip_streak = 0
    dip_candidate_start = None
    dip_base_y = None     # valeur Y au début du dip
    dip_peak_y = None     # max Y atteint pendant le dip

    paused = False
    prev_time = time.time()
    fps_smooth = 0.0

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
                    if hipw is not None:
                        hipw_buf.append(hipw)

            # Si pas assez de données pour lisser, on affiche et continue
            if len(hip_y_buf) >= max(3, SMOOTH_WIN // 2):
                smooth_y = float(median(hip_y_buf))

                # hip width robuste (médiane) => normalisation des seuils
                hipw_med = float(median(hipw_buf)) if len(hipw_buf) > 0 else 200.0
                eps_px = EPS_VEL_RATIO_OF_HIPWIDTH * hipw_med
                A_px = A_MIN_DIP_RATIO_OF_HIPWIDTH * hipw_med

                if prev_smooth_y is not None:
                    vel = smooth_y - prev_smooth_y  # >0: descend / <0: remonte

                    # streak phase descente/remontée
                    if vel > eps_px:
                        down_streak += 1
                    else:
                        down_streak = 0

                    if vel < -eps_px:
                        up_streak += 1
                    else:
                        up_streak = 0

                    # -------- Machine d'états phase --------
                    if state == "idle":
                        if down_streak >= N_DOWN_FRAMES:
                            state = "descending"
                            start_desc_frame = frame_idx - N_DOWN_FRAMES + 1
                            max_smooth_y = smooth_y
                            max_smooth_y_frame = frame_idx
                            print(f"[INFO] Descente détectée à frame={start_desc_frame} (t={start_desc_frame/fps_video:.2f}s)")

                    elif state == "descending":
                        # bottom approx (max Y)
                        if max_smooth_y is None or smooth_y > max_smooth_y:
                            max_smooth_y = smooth_y
                            max_smooth_y_frame = frame_idx

                        if start_desc_frame is not None and (frame_idx - start_desc_frame) >= MIN_DESC_FRAMES:
                            if up_streak >= N_UP_FRAMES:
                                state = "ascending"
                                start_up_frame = frame_idx - N_UP_FRAMES + 1

                                bottom_info = ""
                                if max_smooth_y_frame is not None:
                                    bottom_info = f" | bottom≈frame={max_smooth_y_frame} (t={max_smooth_y_frame/fps_video:.2f}s)"
                                print(f"[OK] Début REMONTÉE à frame={start_up_frame} (t={start_up_frame/fps_video:.2f}s){bottom_info}")

                                # reset dip tracking à l'entrée en montée
                                dip_streak = 0
                                dip_candidate_start = None
                                dip_base_y = None
                                dip_peak_y = None

                    elif state == "ascending":
                        # -------- Détection du dip (redescente) --------
                        # On ignore les X premières frames après le début de montée
                        if (start_up_frame is not None) and (frame_idx - start_up_frame >= X_IGNORE_FRAMES) and (not dip_detected):

                            # dip = mouvement vers le bas pendant la montée => vel > +eps
                            if vel > eps_px:
                                if dip_candidate_start is None:
                                    dip_candidate_start = frame_idx
                                    dip_base_y = prev_smooth_y  # base = juste avant le dip
                                    dip_peak_y = smooth_y

                                dip_streak += 1
                                dip_peak_y = max(dip_peak_y, smooth_y)

                            else:
                                # on vient de sortir d'un dip potentiel => on l'évalue
                                if dip_candidate_start is not None:
                                    amp = (dip_peak_y - dip_base_y) if (dip_peak_y is not None and dip_base_y is not None) else 0.0

                                    # ✅ Condition finale : durée >= K ET amplitude >= A
                                    if dip_streak >= K_MIN_DIP_FRAMES and amp >= A_px:
                                        dip_detected = True
                                        dip_start_frame = dip_candidate_start
                                        dip_end_frame = frame_idx
                                        dip_amp_px = amp
                                        print(
                                            f"[FAULT] Redescente pendant la remontée détectée ! "
                                            f"start={dip_start_frame} (t={dip_start_frame/fps_video:.2f}s), "
                                            f"amp≈{amp:.1f}px (seuil {A_px:.1f}px), "
                                            f"durée={dip_streak} frames"
                                        )

                                    # reset candidat
                                    dip_streak = 0
                                    dip_candidate_start = None
                                    dip_base_y = None
                                    dip_peak_y = None

                prev_smooth_y = smooth_y

                # -------- Overlay --------
                cv2.putText(frame, f"state: {state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if start_up_frame is not None:
                    cv2.putText(frame, f"ASCENT START: frame {start_up_frame}  t={start_up_frame/fps_video:.2f}s",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if state == "ascending":
                    cv2.putText(frame, f"Dip check: ignore first {X_IGNORE_FRAMES} frames, K={K_MIN_DIP_FRAMES}, A={A_px:.1f}px",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if dip_detected:
                    cv2.putText(frame, "FAULT: DIP DURING ASCENT", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # FPS overlay (du programme, pas celui de la vidéo)
            now = time.time()
            dt = now - prev_time
            prev_time = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps
            cv2.putText(frame, f"Live FPS: {fps_smooth:.1f}", (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, "Space: pause | R: reset | Q/Esc: quit", (10, 160),
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
            prev_smooth_y = None
            down_streak = 0
            up_streak = 0
            start_desc_frame = None
            start_up_frame = None
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

            print("[INFO] Reset état/mesures.")

    cap.release()
    cv2.destroyAllWindows()

    # ---- Résultat final clair ----
    print("\n========== RÉSULTAT ==========")
    if start_up_frame is None:
        print("Remontée non détectée -> pas d'analyse dip.")
    else:
        print(f"Début remontée: frame={start_up_frame}  t={start_up_frame/fps_video:.2f}s")

    if dip_detected:
        print(f"FAUTE: redescente pendant la remontée détectée.")
        print(f"  - dip_start: frame={dip_start_frame} t={dip_start_frame/fps_video:.2f}s")
        print(f"  - dip_end:   frame={dip_end_frame} t={dip_end_frame/fps_video:.2f}s")
        print(f"  - amplitude: ~{dip_amp_px:.1f}px (seuil ~{A_MIN_DIP_RATIO_OF_HIPWIDTH*100:.1f}% hipWidth)")
    else:
        print("OK: aucune redescente détectée pendant la remontée.")


if __name__ == "__main__":
    main()
