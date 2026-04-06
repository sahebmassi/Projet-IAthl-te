import os
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .constants import (
    DISK_MODEL_DEFAULT,
    LATERAL_ATHLETE_MODEL_DEFAULT,
    POSE_MODEL_DEFAULT,
    PROCESSING_FPS_DEFAULT,
    SQUAT_BARBELL_MODEL_DEFAULT,
    get_available_pt_models,
    get_lateral_athlete_models,
)
from .video_worker import VideoWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyse powerlifting - PySide6")
        self.resize(1720, 980)
        self.worker = None
        self.video_edits = []
        self.video_buttons = []
        self.video_labels = []
        self._last_movement = "Squat"
        self._build_ui()
        self._build_menu()
        self._apply_style()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right, 3)

        # Charger les fichiers .pt disponibles
        available_models = get_available_pt_models()
        
        self.pose_model_edit = QComboBox()
        self.pose_model_edit.addItems(available_models)
        # Sélectionner le modèle de pose par défaut s'il existe
        pose_default = POSE_MODEL_DEFAULT.split('/')[-1] if POSE_MODEL_DEFAULT else None
        if pose_default and pose_default in available_models:
            self.pose_model_edit.setCurrentText(pose_default)
        
        self.tracking_model_edit = QComboBox()
        self.tracking_model_edit.addItems(available_models)
        # Sélectionner le modèle de suivi par défaut s'il existe
        tracking_default = SQUAT_BARBELL_MODEL_DEFAULT.split('/')[-1] if SQUAT_BARBELL_MODEL_DEFAULT else None
        if tracking_default and tracking_default in available_models:
            self.tracking_model_edit.setCurrentText(tracking_default)
        
        # Charger les modèles latéraux athlète
        lateral_models = get_lateral_athlete_models()
        self.lateral_athlete_model_edit = QComboBox()
        self.lateral_athlete_model_edit.addItems(lateral_models)
        # Sélectionner le modèle latéral athlète par défaut s'il existe
        lateral_default = LATERAL_ATHLETE_MODEL_DEFAULT.split('/')[-1] if LATERAL_ATHLETE_MODEL_DEFAULT else None
        if lateral_default and lateral_default in lateral_models:
            self.lateral_athlete_model_edit.setCurrentText(lateral_default)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.25)

        self.processing_fps_spin = QSpinBox()
        self.processing_fps_spin.setRange(1, 120)
        self.processing_fps_spin.setValue(PROCESSING_FPS_DEFAULT)

        self.mouvement_combo = QComboBox()
        self.mouvement_combo.addItems(
            ["Squat", "Developpe couche", "Souleve de terre"]
        )

        self.vue_combo = QComboBox()
        self.vue_combo.addItems(
            [
                "1 vue",
                "2 vues",
                "3 vues",
            ]
        )

        form_box = QGroupBox("Parametres")
        form = QFormLayout(form_box)
        form.addRow("Type de mouvement", self.mouvement_combo)
        form.addRow("Nombre de vues", self.vue_combo)
        form.addRow("Modele pose", self.pose_model_edit)
        self.tracking_model_label = QLabel("Modele barre")
        form.addRow(self.tracking_model_label, self.tracking_model_edit)
        self.lateral_athlete_label = QLabel("Modele athlete lateral")
        form.addRow(self.lateral_athlete_label, self.lateral_athlete_model_edit)
        form.addRow("Confiance pose", self.conf_spin)
        form.addRow("FPS traitement", self.processing_fps_spin)

        videos_box = QGroupBox("Videos")
        videos_layout = QGridLayout(videos_box)
        for index in range(3):
            label = QLabel()
            edit = QLineEdit()
            button = QPushButton("Parcourir")
            button.clicked.connect(lambda _, idx=index: self.open_video_for_index(idx))
            self.video_labels.append(label)
            self.video_edits.append(edit)
            self.video_buttons.append(button)
            videos_layout.addWidget(label, index, 0)
            videos_layout.addWidget(edit, index, 1)
            videos_layout.addWidget(button, index, 2)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "padding:10px; background:#25272c; border:1px solid #3f4248; border-radius:10px;"
        )

        self.btn_start = QPushButton("Lancer")
        self.btn_pause = QPushButton("Pause / Reprendre")
        self.btn_restart = QPushButton("Recommencer")
        self.btn_stop = QPushButton("Arreter")

        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_restart.clicked.connect(self.restart_analysis)
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.mouvement_combo.currentTextChanged.connect(self.on_movement_changed)
        self.vue_combo.currentTextChanged.connect(self.update_mode_info)
        self.vue_combo.currentTextChanged.connect(self.update_video_fields_visibility)

        left.addWidget(form_box)
        left.addWidget(videos_box)
        left.addWidget(self.info_label)
        left.addWidget(self.btn_start)
        left.addWidget(self.btn_pause)
        left.addWidget(self.btn_restart)
        left.addWidget(self.btn_stop)
        left.addStretch(1)

        self.image_label = QLabel("Aucune video chargee")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(960, 540)
        self.image_label.setStyleSheet(
            "background:#111; border:1px solid #333; border-radius:12px;"
        )
        self.dashboard_text = QTextEdit()
        self.dashboard_text.setReadOnly(True)
        self.dashboard_text.setPlainText("Tableau de bord en attente.")
        self.dashboard_text.setStyleSheet(
            """
            QTextEdit {
                background: #1e1f22;
                color: #f3f3f3;
                border: 1px solid #3f4248;
                border-radius: 8px;
                font-family: monospace;
                font-size: 10px;
            }
            """
        )

        right.addWidget(self.image_label, 3)
        right.addWidget(self.dashboard_text, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Pret")
        self._update_video_labels()
        self.update_video_fields_visibility()
        self.update_mode_info()
        # Rendre invisible le modèle athlète latéral par défaut
        self.lateral_athlete_label.setVisible(False)
        self.lateral_athlete_model_edit.setVisible(False)

    def required_view_count(self):
        current = self.vue_combo.currentText()
        if current.startswith("1 vue"):
            return 1
        if current.startswith("2 vues"):
            return 2
        return 3

    def _movement_model_defaults(self, movement: str):
        if movement == "Squat":
            return SQUAT_BARBELL_MODEL_DEFAULT
        if movement == "Souleve de terre":
            return DISK_MODEL_DEFAULT
        return ""

    def _update_video_labels(self):
        movement = self.mouvement_combo.currentText()
        if movement == "Squat":
            labels = ["Vue face", "Vue laterale 1", "Vue laterale 2"]
        elif movement == "Souleve de terre":
            labels = [
                "Vue face / corps",
                "Vue laterale 1 / disque",
                "Vue laterale 2",
            ]
        else:
            labels = ["Vue principale", "Vue secondaire 1", "Vue secondaire 2"]

        for label_widget, edit, text in zip(self.video_labels, self.video_edits, labels):
            label_widget.setText(text)
            edit.setPlaceholderText(f"Choisir la video : {text}")

    def update_video_fields_visibility(self):
        needed = self.required_view_count()
        for index, (label, edit, button) in enumerate(
            zip(self.video_labels, self.video_edits, self.video_buttons)
        ):
            visible = index < needed
            label.setVisible(visible)
            edit.setVisible(visible)
            button.setVisible(visible)

    def on_movement_changed(self):
        movement = self.mouvement_combo.currentText()
        previous_model = self._movement_model_defaults(self._last_movement)
        new_model = self._movement_model_defaults(movement)

        current_model = self.tracking_model_edit.currentText().strip()
        new_model_name = new_model.split('/')[-1] if new_model else None
        if not current_model or current_model == previous_model.split('/')[-1]:
            if new_model_name and new_model_name in [self.tracking_model_edit.itemText(i) for i in range(self.tracking_model_edit.count())]:
                self.tracking_model_edit.setCurrentText(new_model_name)

        if movement == "Squat":
            self.tracking_model_label.setText("Modele barre")
            self.lateral_athlete_label.setVisible(False)
            self.lateral_athlete_model_edit.setVisible(False)
        elif movement == "Souleve de terre":
            self.tracking_model_label.setText("Modele disque")
            self.lateral_athlete_label.setVisible(True)
            self.lateral_athlete_model_edit.setVisible(True)
        else:
            self.tracking_model_label.setText("Modele suivi")
            self.lateral_athlete_label.setVisible(False)
            self.lateral_athlete_model_edit.setVisible(False)

        self._last_movement = movement
        self._update_video_labels()
        self.update_video_fields_visibility()
        self.update_mode_info()

    def update_mode_info(self):
        mouvement = self.mouvement_combo.currentText()
        vues = self.vue_combo.currentText()
        if mouvement == "Squat":
            txt = (
                f"Mode actuel : {mouvement}\n"
                f"Configuration video : {vues}\n"
                "La vue face pilote les phases du squat.\n"
                "La trajectoire de barre utilise la vue laterale 1 si elle existe.\n"
                "Le modele choisi sert au suivi de barre et le tableau de bord est affiche sous la video."
            )
        elif mouvement == "Souleve de terre":
            txt = (
                f"Mode actuel : {mouvement}\n"
                f"Configuration video : {vues}\n"
                "La vue 1 garde l'analyse du corps et ajoute le suivi visuel de la barre face via barre_face.pt.\n"
                "Si tu choisis 2 ou 3 vues : la vue 2 ajoute en plus la trajectoire via le disque.\n"
                "Le modele choisi sert uniquement au suivi du disque en vue laterale."
            )
        else:
            txt = (
                f"Mode actuel : {mouvement}\n"
                f"Configuration video : {vues}\n"
                "Le moteur developpe couche n'est pas encore implemente."
            )
        self.info_label.setText(txt)

    def _build_menu(self):
        menu = self.menuBar().addMenu("Fichier")
        open_action = QAction("Ouvrir la premiere video", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(lambda: self.open_video_for_index(0))
        menu.addAction(open_action)

        quit_action = QAction("Quitter", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #1e1f22; color: #f3f3f3; }
            QLabel, QGroupBox { color: #f3f3f3; }
            QLineEdit, QDoubleSpinBox, QComboBox {
                background: #2b2d31;
                color: #f3f3f3;
                border: 1px solid #3f4248;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton {
                background: #3a7afe;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background: #5b90ff; }
            QPushButton:pressed { background: #2e65d3; }
            QGroupBox {
                border: 1px solid #3f4248;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 12px;
                background: #25272c;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QStatusBar { background: #25272c; color: #d7d7d7; }
            """
        )

    @Slot()
    def open_video_for_index(self, index: int):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selectionner une video",
            "",
            "Videos (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;Tous les fichiers (*.*)",
        )
        if path:
            self.video_edits[index].setText(path)
            self.statusBar().showMessage(f"Video chargee : {os.path.basename(path)}")

    @Slot()
    def start_analysis(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Analyse en cours",
                "Arrete l'analyse actuelle avant d'en lancer une autre.",
            )
            return

        mouvement = self.mouvement_combo.currentText()
        nb_vues = self.vue_combo.currentText()
        if mouvement == "Developpe couche":
            QMessageBox.information(
                self,
                "Moteur non encore disponible",
                f"Le mode '{mouvement}' est selectionne avec '{nb_vues}', "
                "mais le moteur correspondant n'est pas encore code.",
            )
            return

        needed = self.required_view_count()
        video_paths = []
        for index in range(needed):
            video_path = self.video_edits[index].text().strip()
            if not video_path:
                QMessageBox.warning(
                    self,
                    "Video manquante",
                    f"Choisis la video pour la vue {index + 1}.",
                )
                return
            video_paths.append(video_path)

        pose_model_name = self.pose_model_edit.currentText().strip()
        pose_model_path = str(Path.cwd() / pose_model_name) if pose_model_name and pose_model_name != "Aucun modèle trouvé" else POSE_MODEL_DEFAULT
        tracking_model_name = self.tracking_model_edit.currentText().strip()
        tracking_model_path = str(Path.cwd() / tracking_model_name) if tracking_model_name and tracking_model_name != "Aucun modèle trouvé" else None
        lateral_athlete_model_name = self.lateral_athlete_model_edit.currentText().strip()
        lateral_athlete_model_path = str(Path.cwd() / lateral_athlete_model_name) if lateral_athlete_model_name and lateral_athlete_model_name != "Aucun modèle trouvé" else None
        barbell_model_path = None
        disk_model_path = None

        if mouvement == "Squat":
            barbell_model_path = tracking_model_path
        elif mouvement == "Souleve de terre" and needed > 1:
            disk_model_path = tracking_model_path

        if mouvement == "Squat" and not barbell_model_path:
            QMessageBox.warning(
                self,
                "Modele barre manquant",
                "Le squat a besoin d'un modele de barre.",
            )
            return
        if mouvement == "Souleve de terre" and needed == 1:
            disk_model_path = None
        elif mouvement == "Souleve de terre" and not disk_model_path:
            QMessageBox.warning(
                self,
                "Modele disque manquant",
                "Avec 2 ou 3 vues, le deadlift a besoin d'un modele disque.",
            )
            return

        self.worker = VideoWorker(
            movement=mouvement,
            video_paths=video_paths,
            pose_model_path=pose_model_path,
            barbell_model_path=barbell_model_path,
            disk_model_path=disk_model_path,
            lateral_athlete_model_path=lateral_athlete_model_path,
            processing_fps=float(self.processing_fps_spin.value()),
            conf=float(self.conf_spin.value()),
        )
        self.worker.image_ready.connect(self.update_image)
        self.worker.dashboard_ready.connect(self.update_dashboard)
        self.worker.status_ready.connect(self.statusBar().showMessage)
        self.worker.error_signal.connect(self.show_error)
        self.worker.finished_cleanly.connect(
            lambda: self.statusBar().showMessage("Analyse terminee")
        )
        self.worker.start()
        self.statusBar().showMessage(
            f"Analyse lancee - {mouvement} - {nb_vues} - {self.processing_fps_spin.value()} fps traitement"
        )

    @Slot()
    def toggle_pause(self):
        if self.worker and self.worker.isRunning():
            self.worker.pause_toggle()
            self.statusBar().showMessage("Pause/Reprise demandee")

    @Slot()
    def restart_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_restart()
            self.statusBar().showMessage("Redemarrage demande")

    @Slot()
    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait()
            self.statusBar().showMessage("Analyse arretee")

    @Slot(QImage)
    def update_image(self, qimage: QImage):
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)

    @Slot(str)
    def update_dashboard(self, text: str):
        self.dashboard_text.setPlainText(text)

    @Slot(str)
    def show_error(self, message: str):
        QMessageBox.critical(self, "Erreur", message)
        self.statusBar().showMessage("Erreur")

    def closeEvent(self, event):
        self.stop_analysis()
        super().closeEvent(event)
