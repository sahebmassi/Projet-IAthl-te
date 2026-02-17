import argparse
import math
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from enumIndice import IndiceYolo


def select_video_file():
    """Ouvre une boîte de dialogue pour choisir un fichier vidéo"""
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre principale
    
    filetypes = [
        ("Fichiers vidéo", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
        ("Tous les fichiers", "*.*")
    ]
    
    video_path = filedialog.askopenfilename(
        title="Sélectionnez une vidéo",
        filetypes=filetypes
    )
    
    root.destroy()
    return video_path


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_deg(a: tuple, b: tuple, c: tuple) -> float | None:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    norm_ba = math.hypot(*ba)
    norm_bc = math.hypot(*bc)
    if norm_ba == 0 or norm_bc == 0:
        return None
    cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (norm_ba * norm_bc)
    cos_angle = clamp(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def score_knee_angle(angle: float | None, target: float = 90.0, tolerance: float = 45.0) -> float:
    if angle is None:
        return 0.0
    return clamp(1.0 - abs(angle - target) / tolerance, 0.0, 1.0)


def score_depth(hip: tuple | None, knee: tuple | None) -> float:
    if hip is None or knee is None:
        return 0.0
    return 1.0 if hip[1] >= knee[1] else 0.0


def score_symmetry(angle_left: float | None, angle_right: float | None) -> float:
    if angle_left is None or angle_right is None:
        return 0.0
    diff = abs(angle_left - angle_right)
    if diff <= 15:
        return 1.0
    return clamp(1.0 - (diff - 15.0) / 30.0, 0.0, 1.0)


def compute_squat_score(keypoints: list) -> float:
    left_hip = tuple(keypoints[IndiceYolo.HANCHE_GAUCHE.value])
    right_hip = tuple(keypoints[IndiceYolo.HANCHE_DROITE.value])
    left_knee = tuple(keypoints[IndiceYolo.GENOU_GAUCHE.value])
    right_knee = tuple(keypoints[IndiceYolo.GENOU_DROIT.value])
    left_ankle = tuple(keypoints[IndiceYolo.TALON_GAUCHE.value])
    right_ankle = tuple(keypoints[IndiceYolo.TALON_DROIT.value])

    angle_left = angle_deg(left_hip, left_knee, left_ankle)
    angle_right = angle_deg(right_hip, right_knee, right_ankle)

    knee_score = (score_knee_angle(angle_left) + score_knee_angle(angle_right)) / 2.0
    depth_score = (score_depth(left_hip, left_knee) + score_depth(right_hip, right_knee)) / 2.0
    symmetry_score = score_symmetry(angle_left, angle_right)

    final_score = 100.0 * (0.5 * depth_score + 0.4 * knee_score + 0.1 * symmetry_score)
    return clamp(final_score, 0.0, 100.0)


def average_knee_angle(keypoints: list) -> float | None:
    left_hip = tuple(keypoints[IndiceYolo.HANCHE_GAUCHE.value])
    right_hip = tuple(keypoints[IndiceYolo.HANCHE_DROITE.value])
    left_knee = tuple(keypoints[IndiceYolo.GENOU_GAUCHE.value])
    right_knee = tuple(keypoints[IndiceYolo.GENOU_DROIT.value])
    left_ankle = tuple(keypoints[IndiceYolo.TALON_GAUCHE.value])
    right_ankle = tuple(keypoints[IndiceYolo.TALON_DROIT.value])
    angle_left = angle_deg(left_hip, left_knee, left_ankle)
    angle_right = angle_deg(right_hip, right_knee, right_ankle)
    if angle_left is None or angle_right is None:
        return None
    return (angle_left + angle_right) / 2.0


def hip_below_knee_score(keypoints: list) -> float:
    left_hip = keypoints[IndiceYolo.HANCHE_GAUCHE.value]
    right_hip = keypoints[IndiceYolo.HANCHE_DROITE.value]
    left_knee = keypoints[IndiceYolo.GENOU_GAUCHE.value]
    right_knee = keypoints[IndiceYolo.GENOU_DROIT.value]
    left = 1.0 if left_hip[1] >= left_knee[1] else 0.0
    right = 1.0 if right_hip[1] >= right_knee[1] else 0.0
    return (left + right) / 2.0


def score_from_lowest_point(min_knee_angle: float | None, min_depth_score: float, fault_downward: bool) -> float:
    if min_knee_angle is None:
        return 0.0
    depth_ok = min_depth_score >= 1.0
    if not depth_ok:
        return 0.0
    score = 100.0
    if fault_downward:
        score -= 40.0
    return clamp(score, 0.0, 100.0)


def pick_athlete_id(tracks: dict) -> int | None:
    if not tracks:
        return None
    return max(
        tracks.items(),
        key=lambda item: (item[1]["contact"], item[1]["angle_range"], item[1]["frames"]),
    )[0]


def best_bar_box(result) -> tuple | None:
    if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
        return None
    best = None
    best_area = 0.0
    for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best


def point_in_box(point: list, box: tuple, pad: float = 0.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def bar_contact_score(keypoints: list, person_box: list, bar_box: tuple | None) -> float:
    if bar_box is None or person_box is None:
        return 0.0
    x1, y1, x2, y2 = bar_box
    px1, py1, px2, py2 = person_box
    center_x = (px1 + px2) / 2.0
    within_bar = 1.0 if x1 <= center_x <= x2 else 0.0
    left_wrist = keypoints[IndiceYolo.POIGNET_GAUCHE.value]
    right_wrist = keypoints[IndiceYolo.POIGNET_DROIT.value]
    wrist_in = 1.0 if point_in_box(left_wrist, bar_box, pad=30) or point_in_box(right_wrist, bar_box, pad=30) else 0.0
    return 0.6 * within_bar + 0.4 * wrist_in


def draw_skeleton(image: cv2.Mat, keypoints: list) -> None:
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    for x, y in keypoints:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    for a, b in skeleton:
        xa, ya = keypoints[a]
        xb, yb = keypoints[b]
        cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)


def center_box_id(boxes: list, frame_width: int, frame_height: int) -> int | None:
    if not boxes:
        return None
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    best_idx = None
    best_dist = None
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        bx = (x1 + x2) / 2.0
        by = (y1 + y2) / 2.0
        dist = (bx - center_x) ** 2 + (by - center_y) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def process_video(video_path: str, pose_model_path: str = "yolov8n-pose.pt", 
                  bar_model_path: str = "best.pt", window_width: int = 960, window_height: int = 540) -> int:
    """
    Traite une vidéo avec les modèles YOLO pour analyser le squat.
    
    Args:
        video_path: Chemin vers le fichier vidéo
        pose_model_path: Chemin vers le modèle de pose YOLO
        bar_model_path: Chemin vers le modèle de barre YOLO
        window_width: Largeur de la fenêtre d'affichage
        window_height: Hauteur de la fenêtre d'affichage
    
    Returns:
        0 en cas de succès, 1 en cas d'erreur
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: impossible d'ouvrir la vidéo: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 10.0
    frame_interval = max(1, int(round(fps / 6)))  # ~6 FPS

    pose_model = YOLO(pose_model_path)
    bar_model = YOLO(bar_model_path)
    frame_idx = 0
    movement_active = False
    end_counter = 0
    final_score = None
    rep_count = 0

    start_angle = 150.0
    end_angle = 160.0
    end_hold_frames = max(1, int(round(0.5 * 6)))
    rise_threshold = 3.0
    descent_tolerance = 2.0
    min_knee_angle = None
    min_depth_score = 0.0
    bottom_reached = False
    fault_downward = False
    prev_avg_knee = None
    tracks = {}
    athlete_id = None
    athlete_lock_count = 0
    athlete_missing = 0
    lock_contact_threshold = 0.6
    lock_frames = 3
    max_missing_frames = 8
    window_name = f"Analyse du squat - {video_path.split('/')[-1]}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            bar_result = bar_model.predict(frame, verbose=False)[0]
            bar_box = best_bar_box(bar_result)

            result = pose_model.track(
                frame,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )[0]
            annotated = frame.copy()
            if bar_box is not None:
                x1, y1, x2, y2 = map(int, bar_box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)

            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                ids = []
                if result.boxes is not None and result.boxes.id is not None:
                    ids = result.boxes.id.int().tolist()
                keypoints_list = result.keypoints.xy.tolist()
                for idx, keypoints in enumerate(keypoints_list):
                    if len(keypoints) < 17:
                        continue
                    track_id = ids[idx] if idx < len(ids) else idx
                    person_box = None
                    if result.boxes is not None and len(result.boxes.xyxy) > idx:
                        person_box = result.boxes.xyxy[idx].tolist()
                    contact = bar_contact_score(keypoints, person_box, bar_box)
                    avg_knee = average_knee_angle(keypoints)
                    if avg_knee is None:
                        continue
                    stats = tracks.setdefault(
                        track_id,
                        {"min": avg_knee, "max": avg_knee, "frames": 0, "angle_range": 0.0, "contact": 0.0},
                    )
                    stats["min"] = min(stats["min"], avg_knee)
                    stats["max"] = max(stats["max"], avg_knee)
                    stats["frames"] += 1
                    stats["angle_range"] = stats["max"] - stats["min"]
                    stats["contact"] = max(stats["contact"], contact)

                boxes_list = []
                if result.boxes is not None and len(result.boxes.xyxy) > 0:
                    boxes_list = result.boxes.xyxy.tolist()
                center_idx = center_box_id(boxes_list, frame.shape[1], frame.shape[0])
                candidate_id = None
                if center_idx is not None and center_idx < len(ids):
                    candidate_id = ids[center_idx]
                elif center_idx is not None:
                    candidate_id = center_idx
                else:
                    candidate_id = pick_athlete_id(tracks)

                if athlete_id is None and candidate_id is not None:
                    contact_score = tracks[candidate_id]["contact"]
                    if bar_box is None:
                        athlete_lock_count += 1
                    elif contact_score >= lock_contact_threshold:
                        athlete_lock_count += 1
                    else:
                        athlete_lock_count = 0

                    if athlete_lock_count >= lock_frames:
                        athlete_id = candidate_id
                        athlete_missing = 0

                if athlete_id is not None:
                    athlete_keypoints = None
                    athlete_box = None
                    for idx, keypoints in enumerate(keypoints_list):
                        track_id = ids[idx] if idx < len(ids) else idx
                        if track_id == athlete_id:
                            athlete_keypoints = keypoints
                            if result.boxes is not None and len(result.boxes.xyxy) > idx:
                                x1, y1, x2, y2 = result.boxes.xyxy[idx].tolist()
                                athlete_box = (int(x1), int(y1), int(x2), int(y2))
                            break

                    if athlete_keypoints is not None and len(athlete_keypoints) >= 17:
                        if athlete_box is not None:
                            x1, y1, x2, y2 = athlete_box
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            label_x = x1
                            label_y = max(0, y1 - 10)
                            cv2.putText(
                                annotated,
                                "ATHLETE",
                                (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        draw_skeleton(annotated, athlete_keypoints)
                        avg_knee = average_knee_angle(athlete_keypoints)
                        if avg_knee is not None:
                            if not movement_active and avg_knee < start_angle:
                                movement_active = True
                                end_counter = 0
                                min_knee_angle = avg_knee
                                min_depth_score = hip_below_knee_score(athlete_keypoints)
                                bottom_reached = False
                                fault_downward = False
                                prev_avg_knee = avg_knee

                            if movement_active:
                                min_knee_angle = min(min_knee_angle, avg_knee) if min_knee_angle is not None else avg_knee
                                min_depth_score = max(min_depth_score, hip_below_knee_score(athlete_keypoints))

                                if not bottom_reached and min_knee_angle is not None and avg_knee > min_knee_angle + rise_threshold:
                                    bottom_reached = True

                                if bottom_reached and prev_avg_knee is not None and avg_knee < prev_avg_knee - descent_tolerance:
                                    fault_downward = True

                                if avg_knee > end_angle:
                                    end_counter += 1
                                    if end_counter >= end_hold_frames:
                                        movement_active = False
                                        end_counter = 0
                                        rep_count += 1
                                        final_score = score_from_lowest_point(min_knee_angle, min_depth_score, fault_downward)
                                else:
                                    end_counter = 0
                                prev_avg_knee = avg_knee

                            if movement_active and min_knee_angle is not None:
                                current_score = score_from_lowest_point(min_knee_angle, min_depth_score, fault_downward)
                                cv2.putText(
                                    annotated,
                                    f"Squat en cours: {current_score:.1f}/100",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                        athlete_missing = 0
                    else:
                        athlete_missing += 1
                else:
                    athlete_missing += 1

                if athlete_missing >= max_missing_frames:
                    athlete_id = None
                    athlete_lock_count = 0
                    athlete_missing = 0

            if not movement_active and final_score is not None:
                verdict = "VALIDE" if final_score >= 100.0 else "REFUSE"
                cv2.putText(
                    annotated,
                    f"Score final (rep {rep_count}): {final_score:.1f}/100 - {verdict}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(166) & 0xFF == ord("q"):
                break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    """Fonction principale avec interface de sélection de fichier"""
    
    print("=" * 60)
    print("ANALYSEUR DE SQUAT - YOLOv8")
    print("=" * 60)
    print("\n1. Sélectionnez un fichier vidéo à analyser")
    print("2. L'analyse commencera automatiquement")
    print("3. Appuyez sur 'q' dans la fenêtre vidéo pour quitter")
    print("=" * 60)
    
    # Sélectionner le fichier vidéo
    video_path = select_video_file()
    
    if not video_path:
        print("Aucun fichier sélectionné. Programme terminé.")
        return 0
    
    print(f"\nVidéo sélectionnée: {video_path}")
    print("Chargement des modèles YOLO...")
    
    # Chemins des modèles (peuvent être modifiés si nécessaire)
    pose_model = "yolov8n-pose.pt"
    bar_model = "best.pt"
    
    # Vérifier si les modèles existent
    import os
    if not os.path.exists(pose_model):
        print(f"AVERTISSEMENT: Modèle de pose '{pose_model}' introuvable.")
        print("Tentative de téléchargement...")
        try:
            from ultralytics import download
            download(pose_model)
        except:
            print("Échec du téléchargement. Veuillez vérifier le chemin du modèle.")
    
    if not os.path.exists(bar_model):
        print(f"AVERTISSEMENT: Modèle de barre '{bar_model}' introuvable.")
    
    print("Démarrage de l'analyse...")
    print("Appuyez sur 'q' dans la fenêtre vidéo pour quitter.")
    
    # Lancer le traitement
    return process_video(video_path, pose_model, bar_model)


if __name__ == "__main__":
    raise SystemExit(main())