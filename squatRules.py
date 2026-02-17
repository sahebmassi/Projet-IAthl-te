"""
Système de détection des défauts de squat selon les règles IPF (International Powerlifting Federation)
Détecte 3 défauts principaux:
1. DEPTH_INSUFFISANTE - la hanche n'est pas assez basse par rapport au genou
2. PIED_EN_AVANT - les pieds bougent ou avancent pendant le mouvement
3. REDESCENTE_APRES_MONTEE - redescente pendant la phase de remontée
"""

import math
import numpy as np
from scipy.signal import savgol_filter
from enumIndice import IndiceYolo


class SquatAnalyzer:
    """Analyseur pour détecter les défauts de squat basé sur la logique de test_video.py"""
    
    def __init__(self):
        # Seuils pour la détection (basés sur test_video.py)
        self.START_ANGLE = 150.0  # Angle pour commencer le mouvement
        self.END_ANGLE = 160.0  # Angle pour finir le mouvement
        self.RISE_THRESHOLD = 3.0  # Seuil pour détecter le fond
        self.DESCENT_TOLERANCE = 2.0  # Tolérance pour la redescente
        self.VERTICAL_MOVEMENT_THRESHOLD = 10  # Mouvement vertical min pour détecter une phase
        
        # État du mouvement
        self.movement_active = False
        self.bottom_reached = False
        self.fault_downward = False
        self.depth_insuffisante = False
        
        # Historique
        self.min_knee_angle = None
        self.min_depth_score = 0.0
        self.prev_avg_knee = None
        self.positions_hanches_init = None
        self.frame_defaut = -1
        self.defaut_type = None
        self.rep_count = 0
        self.phase = "attente"
        self.hanche_y_init = None
        self.hanche_y_prev = None
        
    def angle_deg(self, a, b, c):
        """Calcule l'angle en degrés entre 3 points (même logique que test_video.py)"""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        norm_ba = math.hypot(*ba)
        norm_bc = math.hypot(*bc)
        if norm_ba == 0 or norm_bc == 0:
            return None
        cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (norm_ba * norm_bc)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))
    
    def get_average_knee_angle(self, keypoints):
        """Récupère l'angle moyen des genoux"""
        if len(keypoints) < 17:
            return None
        
        try:
            left_hip = tuple(keypoints[IndiceYolo.HANCHE_GAUCHE.value])
            right_hip = tuple(keypoints[IndiceYolo.HANCHE_DROITE.value])
            left_knee = tuple(keypoints[IndiceYolo.GENOU_GAUCHE.value])
            right_knee = tuple(keypoints[IndiceYolo.GENOU_DROIT.value])
            left_ankle = tuple(keypoints[IndiceYolo.TALON_GAUCHE.value])
            right_ankle = tuple(keypoints[IndiceYolo.TALON_DROIT.value])
            
            angle_left = self.angle_deg(left_hip, left_knee, left_ankle)
            angle_right = self.angle_deg(right_hip, right_knee, right_ankle)
            
            if angle_left is None or angle_right is None:
                return None
                
            return (angle_left + angle_right) / 2.0
        except (IndexError, TypeError):
            return None
    
    def hip_below_knee_score(self, keypoints):
        """Calcule si la hanche est en dessous du genou (score de profondeur)"""
        if len(keypoints) < 17:
            return 0.0
        
        try:
            left_hip = keypoints[IndiceYolo.HANCHE_GAUCHE.value]
            right_hip = keypoints[IndiceYolo.HANCHE_DROITE.value]
            left_knee = keypoints[IndiceYolo.GENOU_GAUCHE.value]
            right_knee = keypoints[IndiceYolo.GENOU_DROIT.value]
            
            left = 1.0 if left_hip[1] >= left_knee[1] else 0.0
            right = 1.0 if right_hip[1] >= right_knee[1] else 0.0
            return (left + right) / 2.0
        except (IndexError, TypeError):
            return 0.0
    
    def analyze_frame(self, keypoints, frame_number):
        """
        Analyse un frame et détecte les défauts
        Logique basée sur test_video.py
        Points clés pour le squat:
        - 11: Hanche gauche, 12: Hanche droite
        - 13: Genou gauche, 14: Genou droit
        - 15: Talon gauche, 16: Talon droit
        """
        result = {
            "frame": frame_number,
            "verdict": "EN_COURS",
            "defaut": None,
            "message": "Squat en cours",
            "instant_defaut": -1,
            "avg_angle": None,
            "phase": self.phase
        }
        
        if len(keypoints) < 17:
            return result
        
        # Récupérer l'angle moyen des genoux
        avg_knee = self.get_average_knee_angle(keypoints)
        if avg_knee is None:
            return result
        
        result["avg_angle"] = avg_knee
        
        # Récupérer la position Y des hanches (indices 11 et 12)
        try:
            hanche_gauche_y = keypoints[11][1]
            hanche_droite_y = keypoints[12][1]
            hanche_y_current = (hanche_gauche_y + hanche_droite_y) / 2.0
        except (IndexError, TypeError):
            return result
        
        # Initialiser au premier frame
        if self.hanche_y_init is None:
            self.hanche_y_init = hanche_y_current
            self.hanche_y_prev = hanche_y_current
            return result
        
        # ===== DÉTECTION DE LA PHASE BASÉE SUR LE MOUVEMENT VERTICAL =====
        # Mouvement vers le bas = descente, mouvement vers le haut = remontée
        hanche_mouvement = hanche_y_current - self.hanche_y_prev
        
        if not self.movement_active:
            # Attendre un mouvement vers le bas pour commencer
            if hanche_mouvement > self.VERTICAL_MOVEMENT_THRESHOLD:
                self.movement_active = True
                self.phase = "descente"
                self.min_knee_angle = avg_knee
                self.min_depth_score = self.hip_below_knee_score(keypoints)
                self.bottom_reached = False
                self.fault_downward = False
                self.prev_avg_knee = avg_knee
                result["phase"] = "descente"
        else:
            # En mouvement
            # Mettre à jour les valeurs de profondeur
            if avg_knee is not None:
                self.min_knee_angle = min(self.min_knee_angle, avg_knee) if self.min_knee_angle is not None else avg_knee
                self.min_depth_score = max(self.min_depth_score, self.hip_below_knee_score(keypoints))
            
            # Détection du fond du squat (quand l'angle remonte de RISE_THRESHOLD)
            if not self.bottom_reached and self.min_knee_angle is not None and avg_knee > self.min_knee_angle + self.RISE_THRESHOLD:
                self.bottom_reached = True
                self.phase = "remontee"
                result["phase"] = "remontee"
                
                # Vérifier la profondeur au point le plus bas (DEPTH_INSUFFISANTE)
                if self.min_depth_score < 1.0:
                    self.depth_insuffisante = True
                    self.defaut_type = "DEPTH_INSUFFISANTE"
                    self.frame_defaut = frame_number
                    result["verdict"] = "REFUSE"
                    result["defaut"] = "DEPTH_INSUFFISANTE"
                    result["message"] = f"DEPTH_INSUFFISANTE: Hanche score={self.min_depth_score:.2f} (doit être 1.0). Points 11,12 (hanches) doivent être en Y > points 13,14 (genoux)"
                    result["instant_defaut"] = frame_number
                    self.movement_active = False
                    return result
            
            # Détection de la redescente après le fond (REDESCENTE_APRES_MONTEE)
            if self.bottom_reached and self.prev_avg_knee is not None and avg_knee < self.prev_avg_knee - self.DESCENT_TOLERANCE:
                self.fault_downward = True
                self.defaut_type = "REDESCENTE_APRES_MONTEE"
                self.frame_defaut = frame_number
                result["verdict"] = "REFUSE"
                result["defaut"] = "REDESCENTE_APRES_MONTEE"
                result["message"] = f"REDESCENTE_APRES_MONTEE: Angle a redescendu de {self.prev_avg_knee - avg_knee:.1f}° (tolerance: {self.DESCENT_TOLERANCE}°)"
                result["instant_defaut"] = frame_number
                self.movement_active = False
                return result
            
            # Fin du mouvement
            if hanche_mouvement < -self.VERTICAL_MOVEMENT_THRESHOLD and self.bottom_reached:
                # Mouvement vers le haut maintenu = fin du squat
                self.movement_active = False
                self.rep_count += 1
                self.phase = "fini"
                result["phase"] = "fini"
                
                # Squat valide si pas de défaut
                if not self.depth_insuffisante and not self.fault_downward:
                    result["verdict"] = "VALIDE"
                    result["message"] = "Squat VALIDE ✓"
                else:
                    result["verdict"] = "REFUSE"
                
                return result
            
            # Mettre à jour la phase basée sur le mouvement vertical
            if hanche_mouvement > self.VERTICAL_MOVEMENT_THRESHOLD / 2:
                self.phase = "descente"
            elif hanche_mouvement < -self.VERTICAL_MOVEMENT_THRESHOLD / 2 and self.bottom_reached:
                self.phase = "remontee"
            
            result["phase"] = self.phase
            # Mettre à jour l'angle précédent
            self.prev_avg_knee = avg_knee
        
        self.hanche_y_prev = hanche_y_current
        return result
    
    def get_result(self):
        """Retourne le résultat final du squat"""
        if self.depth_insuffisante:
            return {
                "verdict": "REFUSE",
                "defaut": "DEPTH_INSUFFISANTE",
                "message": "Le squat est REFUSÉ: Profondeur insuffisante",
                "frame_defaut": self.frame_defaut
            }
        
        if self.fault_downward:
            return {
                "verdict": "REFUSE",
                "defaut": "REDESCENTE_APRES_MONTEE",
                "message": "Le squat est REFUSÉ: Redescente pendant la remontée",
                "frame_defaut": self.frame_defaut
            }
        
        return {
            "verdict": "VALIDE",
            "defaut": None,
            "message": "Le squat est VALIDE ✓",
            "frame_defaut": -1
        }
    
    def reset(self):
        """Réinitialise l'analyseur pour un nouveau squat"""
        self.movement_active = False
        self.bottom_reached = False
        self.fault_downward = False
        self.depth_insuffisante = False
        self.min_knee_angle = None
        self.min_depth_score = 0.0
        self.prev_avg_knee = None
        self.frame_defaut = -1
        self.defaut_type = None


class SquatAnalyzerOffline:
    """
    Analyseur OFFLINE robuste pour le squat.
    Basé sur l'analyse COMPLÈTE de la vidéo après lecture.
    
    Étapes:
    1. Lissage des keypoints (Savitzky-Golay)
    2. Détermination du "bottom" (point de plus grande flexion)
    3. Détection des phases de manière robuste
    4. Détection des 3 défauts avec persistance
    """
    
    def __init__(self, fps=30, view_id="center"):
        self.fps = fps
        self.view_id = view_id
        
        # Seuils (à ajuster selon les tests)
        self.DEPTH_THRESHOLD = 1.0  # Hanche doit être >= genou en Y
        self.BOUNCE_PERSIST_FRAMES = 5  # Nombre de frames pour confirmer un bounce
        self.BOUNCE_AMPLITUDE_RATIO = 0.1  # 10% de l'amplitude totale
        
    def smooth_signal(self, signal, window_length=11, polyorder=3):
        """Lisse un signal avec Savitzky-Golay."""
        if len(signal) < window_length:
            return np.array(signal)
        # Assurer window_length est impair
        if window_length % 2 == 0:
            window_length += 1
        try:
            return savgol_filter(signal, window_length, polyorder)
        except:
            return np.array(signal)
    
    def extract_signal_from_keypoints(self, keypoints_series, joint_indices):
        """
        Extrait un signal (ex: positions Y de la hanche) à partir des keypoints.
        
        Args:
            keypoints_series: list of frames, each frame is list of 17 keypoints (x,y)
            joint_indices: list of indices (ex: [11, 12] pour les 2 hanches)
        
        Returns:
            np.array of values (ex: Y positions moyennes)
        """
        signal = []
        for frame_keypoints in keypoints_series:
            if len(frame_keypoints) < 17:
                signal.append(np.nan)
                continue
            
            values = []
            for idx in joint_indices:
                if idx < len(frame_keypoints) and frame_keypoints[idx] is not None:
                    values.append(frame_keypoints[idx][1])  # Y coordinate
            
            if values:
                signal.append(np.nanmean(values))
            else:
                signal.append(np.nan)
        
        return np.array(signal, dtype=float)
    
    def find_bottom_frame(self, keypoints_series):
        """
        Trouve le frame du "bottom" du squat (flexion max).
        
        Logique:
        - Calculer la position Y moyenne des hanches
        - Lisser le signal
        - Trouver le max de Y (plus bas) après un certain point
        - Retourner l'index du frame
        """
        hanche_y = self.extract_signal_from_keypoints(keypoints_series, [11, 12])
        
        # Interpoler les NaN
        mask = ~np.isnan(hanche_y)
        if not np.any(mask):
            return len(keypoints_series) // 2  # Fallback au milieu
        
        hanche_y_interp = hanche_y.copy()
        indices = np.arange(len(hanche_y))
        hanche_y_interp[~mask] = np.interp(indices[~mask], indices[mask], hanche_y[mask])
        
        # Lisser
        hanche_y_smooth = self.smooth_signal(hanche_y_interp, window_length=11, polyorder=3)
        
        # Trouver le bottom (max Y = position la plus basse)
        # Éviter les premiers et derniers frames (bruit)
        start_idx = max(1, len(hanche_y_smooth) // 4)
        end_idx = min(len(hanche_y_smooth) - 1, 3 * len(hanche_y_smooth) // 4)
        
        bottom_idx = start_idx + np.argmax(hanche_y_smooth[start_idx:end_idx])
        
        return int(bottom_idx), hanche_y_smooth
    
    def detect_phases(self, keypoints_series):
        """
        Détecte les phases de mouvement (descente/remontée).
        
        Retourne:
            (phase_labels, bottom_frame_idx)
            phase_labels: array of "descente", "remontee" ou "unknown"
        """
        bottom_idx, hanche_y_smooth = self.find_bottom_frame(keypoints_series)
        
        phase_labels = []
        for i in range(len(keypoints_series)):
            if i <= bottom_idx:
                phase_labels.append("descente")
            else:
                phase_labels.append("remontee")
        
        return np.array(phase_labels), bottom_idx
    
    def check_depth(self, keypoints_series, bottom_idx, tolerance=10):
        """
        Vérifie si la profondeur est suffisante au bottom.
        
        Critère IPF: hanche doit être au-dessous du genou (hip_y >= knee_y)
        
        Returns:
            (is_valid, confidence_score)
        """
        window_start = max(0, bottom_idx - tolerance)
        window_end = min(len(keypoints_series), bottom_idx + tolerance + 1)
        
        valid_count = 0
        total_count = 0
        
        for frame_idx in range(window_start, window_end):
            kpts = keypoints_series[frame_idx]
            if len(kpts) < 17:
                continue
            
            total_count += 1
            
            hanche_left_y = kpts[11][1] if kpts[11] is not None else None
            hanche_right_y = kpts[12][1] if kpts[12] is not None else None
            genou_left_y = kpts[13][1] if kpts[13] is not None else None
            genou_right_y = kpts[14][1] if kpts[14] is not None else None
            
            # Vérifier chaque côté
            if hanche_left_y is not None and genou_left_y is not None:
                if hanche_left_y >= genou_left_y:
                    valid_count += 1
            
            if hanche_right_y is not None and genou_right_y is not None:
                if hanche_right_y >= genou_right_y:
                    valid_count += 1
        
        if total_count == 0:
            return False, 0.0
        
        confidence = valid_count / (2 * total_count)  # Max 2 per frame (left+right)
        is_valid = confidence >= self.DEPTH_THRESHOLD
        
        return is_valid, confidence
    
    def check_bounce(self, keypoints_series, phase_labels, bottom_idx):
        """
        Détecte une redescente pendant la remontée (bounce).
        
        Logique:
        - Après le bottom, la hanche Y devrait diminuer (remonter)
        - Si Y augmente (redescend) pendant N frames consecutives avec amplitude >= seuil
        - C'est un bounce/redescente
        
        Returns:
            (has_bounce, bounce_frame_idx, bounce_amplitude)
        """
        hanche_y = self.extract_signal_from_keypoints(keypoints_series, [11, 12])
        
        # Interpoler NaN
        mask = ~np.isnan(hanche_y)
        if not np.any(mask):
            return False, -1, 0.0
        
        hanche_y_interp = hanche_y.copy()
        indices = np.arange(len(hanche_y))
        hanche_y_interp[~mask] = np.interp(indices[~mask], indices[mask], hanche_y[mask])
        
        # Lisser
        hanche_y_smooth = self.smooth_signal(hanche_y_interp, window_length=11, polyorder=3)
        
        # Amplitude totale du mouvement
        amplitude_total = np.nanmax(hanche_y_smooth) - np.nanmin(hanche_y_smooth)
        bounce_amplitude_threshold = amplitude_total * self.BOUNCE_AMPLITUDE_RATIO
        
        # Chercher redescente après le bottom
        for i in range(bottom_idx + 5, len(hanche_y_smooth) - self.BOUNCE_PERSIST_FRAMES):
            # Vérifier si les N frames suivantes sont en redescente
            redescente_count = 0
            max_bounce_amplitude = 0
            
            for j in range(i, min(i + self.BOUNCE_PERSIST_FRAMES, len(hanche_y_smooth) - 1)):
                dy = hanche_y_smooth[j + 1] - hanche_y_smooth[j]
                if dy > 0:  # Y augmente = redescend
                    redescente_count += 1
                    max_bounce_amplitude = max(max_bounce_amplitude, abs(dy))
            
            if redescente_count >= self.BOUNCE_PERSIST_FRAMES - 1 and max_bounce_amplitude >= bounce_amplitude_threshold:
                return True, i, max_bounce_amplitude
        
        return False, -1, 0.0
    

    
    def analyze(self, keypoints_series):
        """
        Analyse complète de la séquence de keypoints.
        
        Args:
            keypoints_series: list of frames, each frame is list of 17 keypoints
        
        Returns:
            {
                "verdict": "VALIDE" ou "REFUSE",
                "defaut": None ou "DEPTH_INSUFFISANTE" ou "REDESCENTE_APRES_MONTEE",
                "message": str,
                "frame_defaut": int,
                "bottom_frame": int,
                "details": {...}
            }
        """
        if len(keypoints_series) < 10:
            return {
                "verdict": "INVALIDE",
                "defaut": "INSUFFICIENT_FRAMES",
                "message": "Trop peu de frames pour analyser",
                "frame_defaut": -1,
                "bottom_frame": -1,
                "details": {}
            }
        
        # Étape 1: Déterminer le bottom
        bottom_idx, _ = self.find_bottom_frame(keypoints_series)
        
        # Étape 2: Déterminer les phases
        phase_labels, _ = self.detect_phases(keypoints_series)
        
        # Étape 3: Vérifier la profondeur
        depth_valid, depth_confidence = self.check_depth(keypoints_series, bottom_idx)
        
        # Étape 4: Vérifier le bounce
        has_bounce, bounce_frame, bounce_amplitude = self.check_bounce(keypoints_series, phase_labels, bottom_idx)
        
        # Déterminer le verdict
        result = {
            "bottom_frame": bottom_idx,
            "phase_labels": phase_labels,
            "details": {
                "depth_valid": depth_valid,
                "depth_confidence": float(depth_confidence),
                "has_bounce": has_bounce,
                "bounce_frame": bounce_frame,
                "bounce_amplitude": float(bounce_amplitude),
                "view_id": self.view_id
            }
        }
        
        # Appliquer les règles IPF (ordre de priorité)
        if not depth_valid:
            result["verdict"] = "REFUSE"
            result["defaut"] = "DEPTH_INSUFFISANTE"
            result["message"] = f"DEPTH_INSUFFISANTE: Hanche n'est pas assez basse au bottom (confidence: {depth_confidence:.2f}). La hanche doit être au-dessous du genou."
            result["frame_defaut"] = bottom_idx
        
        elif has_bounce:
            result["verdict"] = "REFUSE"
            result["defaut"] = "REDESCENTE_APRES_MONTEE"
            result["message"] = f"REDESCENTE_APRES_MONTEE: L'athlète a redescendu pendant la remontée au frame {bounce_frame} (amplitude: {bounce_amplitude:.1f}px)"
            result["frame_defaut"] = bounce_frame
        
        else:
            result["verdict"] = "VALIDE"
            result["defaut"] = None
            result["message"] = "Squat VALIDE ✓"
            result["frame_defaut"] = -1
        
        return result


def analyze_sequence(keypoints_series, fps=30, view_id="center"):
    """
    Fonction wrapper pour analyser une séquence complète de keypoints.
    
    Args:
        keypoints_series: list of frames, each frame is list of 17 keypoints (x, y)
        fps: frames per second (pour la synchronisation multi-vue si besoin)
        view_id: identifiant de la vue ("center", "left", "right")
    
    Returns:
        dict avec le verdict et détails de l'analyse
    """
    analyzer = SquatAnalyzerOffline(fps=fps, view_id=view_id)
    return analyzer.analyze(keypoints_series)
