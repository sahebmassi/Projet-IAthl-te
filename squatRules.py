"""
Système de détection des défauts de squat selon les règles IPF (International Powerlifting Federation)
Détecte 3 défauts principaux:
1. DEPTH_INSUFFISANTE - la hanche n'est pas assez basse par rapport au genou
2. PIED_EN_AVANT - les pieds bougent ou avancent pendant le mouvement
3. REDESCENTE_APRES_MONTEE - redescente pendant la phase de remontée
"""

import math
from enumIndice import IndiceYolo


class SquatAnalyzer:
    """Analyseur pour détecter les défauts de squat basé sur la logique de test_video.py"""
    
    def __init__(self):
        # Seuils pour la détection (basés sur test_video.py)
        self.START_ANGLE = 150.0  # Angle pour commencer le mouvement
        self.END_ANGLE = 160.0  # Angle pour finir le mouvement
        self.RISE_THRESHOLD = 3.0  # Seuil pour détecter le fond
        self.DESCENT_TOLERANCE = 2.0  # Tolérance pour la redescente
        self.PIED_MOVEMENT_THRESHOLD = 20  # Pixels de tolérance pour le mouvement des pieds
        self.VERTICAL_MOVEMENT_THRESHOLD = 10  # Mouvement vertical min pour détecter une phase
        
        # État du mouvement
        self.movement_active = False
        self.bottom_reached = False
        self.fault_downward = False
        self.pied_en_avant = False
        self.depth_insuffisante = False
        
        # Historique
        self.min_knee_angle = None
        self.min_depth_score = 0.0
        self.prev_avg_knee = None
        self.positions_pieds_init = None
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
    
    def get_foot_positions(self, keypoints):
        """Récupère les positions des pieds"""
        if len(keypoints) < 17:
            return None, None
        
        try:
            left_ankle = keypoints[IndiceYolo.TALON_GAUCHE.value]
            right_ankle = keypoints[IndiceYolo.TALON_DROIT.value]
            return left_ankle, right_ankle
        except (IndexError, TypeError):
            return None, None
    
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
        
        # Initialiser les positions au premier frame
        if self.positions_pieds_init is None:
            self.positions_pieds_init = self.get_foot_positions(keypoints)
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
            # Vérifier le mouvement des pieds (PIED_EN_AVANT)
            if not self.pied_en_avant:
                left_ankle, right_ankle = self.get_foot_positions(keypoints)
                left_init, right_init = self.positions_pieds_init
                
                if left_ankle and right_ankle and left_init and right_init:
                    left_move = abs(left_ankle[0] - left_init[0])
                    right_move = abs(right_ankle[0] - right_init[0])
                    
                    if left_move > self.PIED_MOVEMENT_THRESHOLD or right_move > self.PIED_MOVEMENT_THRESHOLD:
                        self.pied_en_avant = True
                        self.defaut_type = "PIED_EN_AVANT"
                        self.frame_defaut = frame_number
                        result["verdict"] = "REFUSE"
                        result["defaut"] = "PIED_EN_AVANT"
                        result["message"] = f"PIED_EN_AVANT: Talon gauche bouge de {left_move:.0f}px, talon droit de {right_move:.0f}px (seuil: {self.PIED_MOVEMENT_THRESHOLD}px)"
                        result["instant_defaut"] = frame_number
                        self.movement_active = False
                        return result
            
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
                if not self.depth_insuffisante and not self.fault_downward and not self.pied_en_avant:
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
        
        if self.pied_en_avant:
            return {
                "verdict": "REFUSE",
                "defaut": "PIED_EN_AVANT",
                "message": "Le squat est REFUSÉ: Les pieds ont bougé",
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
        self.pied_en_avant = False
        self.depth_insuffisante = False
        self.min_knee_angle = None
        self.min_depth_score = 0.0
        self.prev_avg_knee = None
        self.positions_pieds_init = None
        self.frame_defaut = -1
        self.defaut_type = None
