"""
Futsal Team Analysis System - Simplified Version
Análisis de estadísticas de equipos de futsal mediante cámara fija lateral.
Detección y clasificación de equipos sin tracking de IDs persistentes.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import time
import subprocess
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ===================== CONFIGURATION =====================

@dataclass
class Config:
    """Configuración centralizada del sistema de análisis."""
    model_name: str = "yolo11n.pt"
    board_width: int = 700
    board_height: int = 350
    
    player_class_id: int = 0
    ball_class_id: int = 32
    
    min_players_for_kmeans: int = 6
    
    yolo_conf_threshold: float = 0.3
    
    stream_read_retries: int = 30
    stream_read_delay: float = 0.3
    yt_dlp_timeout: int = 30


config = Config()


# ===================== YOUTUBE STREAM =====================

def open_youtube_stream(url: str, config: Config) -> Optional[cv2.VideoCapture]:
    """Abre un video de YouTube como OpenCV VideoCapture."""
    formats = ["best[height<=1080]", "best[height<=720]", "best[height<=480]", "best[height<=360]", "best"]
    
    for fmt in formats:
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", fmt, "--get-url", url],
                capture_output=True,
                text=True,
                timeout=config.yt_dlp_timeout
            )
            
            if result.returncode != 0:
                logger.warning(f"yt-dlp falló para {fmt}: {result.stderr.strip()[:120]}")
                continue
            
            stream_url = result.stdout.strip().split("\n")[0]
            if not stream_url:
                continue
            
            cap = cv2.VideoCapture(stream_url)
            if cap.isOpened():
                logger.info(f"Stream abierto exitosamente ({fmt})")
                return cap
            cap.release()
            
        except FileNotFoundError:
            logger.error("yt-dlp no encontrado. Instalar con: pip install yt-dlp")
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"yt-dlp timeout para {fmt}")
    
    logger.error("Falló al abrir stream con cualquier formato")
    return None


# ===================== FIELD CALIBRATOR =====================

class FieldCalibrator:
    """Sistema de calibración por 6 puntos independientes para definir los límites del campo."""
    
    def __init__(self, frame: np.ndarray):
        """Inicializa el calibrador con 6 puntos independientes."""
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        
        self.padding = max(200, int(self.h * 0.3))
        self.canvas_w = self.w + 2 * self.padding
        self.canvas_h = self.h + 2 * self.padding
        self.offset_x = self.padding
        self.offset_y = self.padding
        
        margin_x = int(self.w * 0.1)
        margin_y = int(self.h * 0.15)
        cx = self.w // 2
        
        self.points = np.array([
            [margin_x, margin_y],
            [cx, margin_y],
            [self.w - margin_x, margin_y],
            [self.w - margin_x, self.h - margin_y],
            [cx, self.h - margin_y],
            [margin_x, self.h - margin_y],
        ], dtype=np.float32)
        
        self.dragging = False
        self.active_point = None
        self.WIN = "Calibración de Campo"
    
    def draw_frame(self) -> np.ndarray:
        """Dibuja el frame con el polígono y 6 puntos de control."""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), (80, 80, 80), dtype=np.uint8)
        canvas[self.offset_y:self.offset_y + self.h, self.offset_x:self.offset_x + self.w] = self.frame.copy()
        
        pts = (self.points + np.array([self.offset_x, self.offset_y])).astype(np.int32)
        
        overlay = canvas.copy()
        cv2.rectangle(overlay, (self.offset_x, self.offset_y), 
                     (self.offset_x + self.w, self.offset_y + self.h), (0, 0, 0), -1)
        cv2.fillPoly(overlay, [pts], (50, 100, 50))
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
        
        cv2.rectangle(canvas, (self.offset_x, self.offset_y), 
                     (self.offset_x + self.w - 1, self.offset_y + self.h - 1), (200, 200, 200), 2)
        
        for i in range(len(pts)):
            next_i = (i + 1) % len(pts)
            cv2.line(canvas, tuple(pts[i]), tuple(pts[next_i]), (0, 255, 0), 3)
        
        handle_size = 12
        labels = ["TL", "CT", "TR", "BR", "CB", "BL"]
        
        for idx, (pt, label) in enumerate(zip(pts, labels)):
            x, y = int(pt[0]), int(pt[1])
            
            if self.active_point == idx and self.dragging:
                cv2.circle(canvas, (x, y), handle_size, (0, 165, 255), -1)
            else:
                cv2.circle(canvas, (x, y), handle_size, (0, 255, 255), -1)
            
            cv2.circle(canvas, (x, y), handle_size, (255, 255, 255), 3)
            cv2.putText(canvas, str(idx), (x-5, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(canvas, label, (x-15, y-handle_size-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        info_text = "CLIC + ARRASTRAR puntos | ESPACIO: confirmar | R: resetear | ESC: cancelar"
        cv2.putText(canvas, info_text, (self.offset_x + 10, self.offset_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return canvas
    
    def get_point_index(self, x: int, y: int):
        """Retorna el índice del punto más cercano si está dentro del threshold."""
        threshold = 25
        for idx, pt in enumerate(self.points):
            dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
            if dist < threshold:
                return idx
        return None
    
    def update_point(self, point_idx: int, x: int, y: int):
        """Actualiza la posición de un punto de control."""
        self.points[point_idx] = [x, y]
    
    def calibrate(self) -> np.ndarray:
        """Ejecutar la interfaz de calibración."""
        print("\n" + "="*70)
        print("CALIBRACIÓN DE CAMPO - FUTSAL")
        print("="*70)
        print("\nPUNTOS A CALIBRAR (numerados):")
        print("  0 = TL (Top-Left)        - Esquina arriba-izquierda")
        print("  1 = CT (Central Top)     - Centro línea superior")
        print("  2 = TR (Top-Right)       - Esquina arriba-derecha")
        print("  3 = BR (Bottom-Right)    - Esquina abajo-derecha")
        print("  4 = CB (Central Bottom)  - Centro línea inferior")
        print("  5 = BL (Bottom-Left)     - Esquina abajo-izquierda")
        print("\nCONTROLES:")
        print("  - CLIC + ARRASTRAR en puntos para ajustar")
        print("  - ESPACIO para confirmar")
        print("  - R para resetear")
        print("  - ESC para cancelar")
        print("="*70 + "\n")
        
        cv2.namedWindow(self.WIN)
        
        def mouse_event(event, x, y, flags, param):
            frame_x = x - self.offset_x
            frame_y = y - self.offset_y
            
            if event == cv2.EVENT_LBUTTONDOWN:
                point_idx = self.get_point_index(frame_x, frame_y)
                if point_idx is not None:
                    self.dragging = True
                    self.active_point = point_idx
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging and self.active_point is not None:
                    self.update_point(self.active_point, frame_x, frame_y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False
                self.active_point = None
        
        cv2.setMouseCallback(self.WIN, mouse_event)
        
        print("Ventana abierta. Mueve los 6 puntos amarillos.\n")
        
        while True:
            img = self.draw_frame()
            cv2.imshow(self.WIN, img)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord(' '):
                print("[OK] Calibración confirmada\n")
                break
            elif key == ord('r') or key == ord('R'):
                margin_x = int(self.w * 0.1)
                margin_y = int(self.h * 0.15)
                cx = self.w // 2
                self.points = np.array([
                    [margin_x, margin_y],
                    [cx, margin_y],
                    [self.w - margin_x, margin_y],
                    [self.w - margin_x, self.h - margin_y],
                    [cx, self.h - margin_y],
                    [margin_x, self.h - margin_y],
                ], dtype=np.float32)
                print("[RESET] Puntos restaurados a valores por defecto\n")
            elif key == 27:
                print("[CANCELAR] Calibración cancelada\n")
                cv2.destroyWindow(self.WIN)
                return np.array([])
        
        cv2.destroyWindow(self.WIN)
        return self.points


# ===================== FIELD VALIDATOR =====================

class FieldValidator:
    """Valida si los jugadores están dentro del campo calibrado."""
    
    def __init__(self, field_rect: np.ndarray):
        """Inicializar con 6 puntos del campo."""
        self.field_rect = field_rect.astype(np.float32)
    
    def is_within_field(self, pt: np.ndarray) -> bool:
        """Verificar si un punto está dentro del polígono del campo."""
        try:
            result = cv2.pointPolygonTest(self.field_rect, tuple(pt), False)
            return result >= 0
        except:
            return True
    
    def filter_detections(self, detections, feet_coords) -> np.ndarray:
        """Filtrar detecciones por ubicación dentro del campo."""
        valid_mask = []
        
        for i, foot in enumerate(feet_coords):
            foot_valid = self.is_within_field(foot)
            
            bbox = detections.xyxy[i]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            center = np.array([center_x, center_y])
            center_valid = self.is_within_field(center)
            
            is_valid = foot_valid or center_valid
            valid_mask.append(is_valid)
        
        return np.array(valid_mask)


# ===================== TEAM CLASSIFIER =====================

class TeamClassifier:
    """Separa jugadores en dos equipos usando K-Means en color de camiseta (HSV)."""

    def __init__(self, n_clusters: int = 3, n_init: int = 20):
        """Inicializar el clasificador."""
        from sklearn.cluster import KMeans
        
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        self.trained: bool = False
        self.ref_label: Optional[int] = None
        self.color_features: Optional[np.ndarray] = None

    @staticmethod
    def _safe_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Recortar una región del frame de forma segura."""
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2]

    def get_jersey_color_features(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extraer características de color HSV de la camiseta."""
        x1, y1, x2, y2 = map(int, bbox)
        jersey_h = int((y2 - y1) * 0.4)
        crop = self._safe_crop(frame, x1, y1, x2, y1 + jersey_h)
        
        if crop.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation_mask = hsv[:, :, 1] > 30
        
        if saturation_mask.sum() < 5:
            return np.mean(hsv, axis=(0, 1)).astype(np.float32)
        
        filtered_hsv = hsv[saturation_mask]
        features = np.mean(filtered_hsv, axis=0).astype(np.float32)
        features[0] = np.clip(features[0], 0, 180)
        
        return features

    def train_teams(self, frame: np.ndarray, detections, config: Config) -> bool:
        """Entrenar clasificador K-Means en colores de camiseta."""
        if len(detections) < config.min_players_for_kmeans:
            logger.debug(f"[TRAIN] No suficientes jugadores ({len(detections)}/{config.min_players_for_kmeans})")
            return False
        
        try:
            logger.info(f"[TRAIN] Extrayendo características de color de {len(detections)} jugadores...")
            color_features = np.array(
                [self.get_jersey_color_features(frame, b) for b in detections.xyxy],
                dtype=np.float32
            )
            
            self.color_features = color_features
            hue_var = np.var(color_features[:, 0])
            
            if hue_var < 50:
                logger.warning("[TRAIN] Varianza de color insuficiente para clasificación")
                return False
            
            logger.info("[TRAIN] Entrenando K-Means...")
            self.kmeans.fit(color_features)
            self.trained = True
            
            counts = np.bincount(self.kmeans.labels_)
            self.ref_label = int(np.argmin(counts))
            
            logger.info(f"[TRAIN] K-Means entrenado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error entrenando equipos: {e}")
            return False

    def predict_team(self, frame: np.ndarray, bbox: np.ndarray) -> int:
        """Predecir equipo para un jugador."""
        if not self.trained:
            return 0
        
        try:
            features = self.get_jersey_color_features(frame, bbox)
            label = int(self.kmeans.predict(features.reshape(1, -1))[0])
            
            if label == self.ref_label:
                return -1
            
            team_labels = [l for l in range(self.kmeans.n_clusters) if l != self.ref_label]
            return team_labels.index(label) if label in team_labels else 0
            
        except Exception as e:
            logger.warning(f"Error prediciendo equipo: {e}")
            return 0


# ===================== SIMPLE BALL TRACKER =====================

class SimpleBallTracker:
    """
    Rastreador simple pero robusto del balón.
    Usa filtrado por movimiento y suavizado exponencial.
    """
    
    def __init__(self, max_speed: float = 200.0, smoothing_alpha: float = 0.5):
        """
        Inicializar rastreador de balón.
        
        Args:
            max_speed: Velocidad máxima del balón en píxeles por frame
            smoothing_alpha: Factor de suavizado exponencial (0-1)
        """
        self.last_pos: Optional[np.ndarray] = None
        self.last_smoothed_pos: Optional[np.ndarray] = None
        self.max_speed = max_speed
        self.smoothing_alpha = smoothing_alpha
        self.frames_tracked = 0
        self.lost_frames = 0
        self.max_lost_frames = 3
    
    def update(self, current_detections: Optional[List[np.ndarray]]) -> Optional[np.ndarray]:
        """
        Actualizar posición del balón.
        
        Args:
            current_detections: Lista de posiciones detectadas del balón o None
            
        Returns:
            Posición suavizada del balón o None
        """
        selected_pos = None
        
        if current_detections is not None and len(current_detections) > 0:
            if len(current_detections) == 1:
                selected_pos = current_detections[0].copy()
            else:
                # Múltiples detecciones: escoger la más consistente con historial
                if self.last_pos is not None:
                    distances = [np.linalg.norm(det - self.last_pos) for det in current_detections]
                    valid_mask = np.array(distances) < self.max_speed
                    
                    if np.sum(valid_mask) > 0:
                        valid_indices = np.where(valid_mask)[0]
                        best_idx = valid_indices[np.argmin([distances[i] for i in valid_indices])]
                        selected_pos = current_detections[best_idx].copy()
                    else:
                        best_idx = np.argmin(distances)
                        selected_pos = current_detections[best_idx].copy()
                else:
                    selected_pos = current_detections[0].copy()
            
            # Aplicar suavizado exponencial
            if self.last_smoothed_pos is not None:
                smoothed = (self.smoothing_alpha * selected_pos + 
                           (1 - self.smoothing_alpha) * self.last_smoothed_pos)
            else:
                smoothed = selected_pos.copy()
            
            self.last_pos = selected_pos
            self.last_smoothed_pos = smoothed
            self.frames_tracked += 1
            self.lost_frames = 0
            
            return smoothed
        else:
            # Sin detección
            self.lost_frames += 1
            
            if self.lost_frames > self.max_lost_frames:
                self.last_pos = None
                self.last_smoothed_pos = None
                return None
            else:
                if self.last_smoothed_pos is not None:
                    return self.last_smoothed_pos.copy()
                else:
                    return None


# ===================== TACTICAL BOARD =====================

class TacticalBoard:
    """Renderiza un diagrama overhead del campo de futsal."""

    TEAM_COLORS = ((255, 80, 80), (80, 255, 255))
    PITCH_COLOR = (34, 139, 34)

    def __init__(self, width: int, height: int):
        """Inicializar tablero táctico."""
        self.width: int = width
        self.height: int = height

    def _draw_pitch(self, board: np.ndarray) -> None:
        """Dibujar líneas del campo."""
        line_color = (255, 255, 255)
        center_x = self.width // 2
        center_y = self.height // 2
        
        cv2.rectangle(board, (0, 0), (self.width - 1, self.height - 1), line_color, 3)
        cv2.line(board, (center_x, 0), (center_x, self.height), line_color, 2)
        
        cv2.circle(board, (center_x, center_y), 5, line_color, -1)
        cv2.circle(board, (center_x, center_y), 60, line_color, 2)
        
        pen_box_w, pen_box_h = 120, 200
        goal_box_w, goal_box_h = 60, 100
        
        left_pen_y1 = center_y - pen_box_h // 2
        left_pen_y2 = center_y + pen_box_h // 2
        left_goal_y1 = center_y - goal_box_h // 2
        left_goal_y2 = center_y + goal_box_h // 2
        
        cv2.rectangle(board, (0, left_pen_y1), (pen_box_w, left_pen_y2), line_color, 2)
        cv2.rectangle(board, (0, left_goal_y1), (goal_box_w, left_goal_y2), line_color, 2)
        cv2.circle(board, (int(pen_box_w * 0.6), center_y), 3, line_color, -1)
        
        right_pen_y1 = center_y - pen_box_h // 2
        right_pen_y2 = center_y + pen_box_h // 2
        right_goal_y1 = center_y - goal_box_h // 2
        right_goal_y2 = center_y + goal_box_h // 2
        
        cv2.rectangle(board, (self.width - pen_box_w, right_pen_y1), 
                      (self.width, right_pen_y2), line_color, 2)
        cv2.rectangle(board, (self.width - goal_box_w, right_goal_y1), 
                      (self.width, right_goal_y2), line_color, 2)
        cv2.circle(board, (self.width - int(pen_box_w * 0.6), center_y), 3, line_color, -1)

    def draw_state(self, players: List[Tuple[np.ndarray, int]], 
                   ball_mapped: Optional[np.ndarray]) -> np.ndarray:
        """
        Dibujar estado actual del partido.
        
        Args:
            players: Lista de (posición, team_id)
            ball_mapped: Posición de la pelota o None
            
        Returns:
            Imagen del tablero renderizada
        """
        board = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        board[:] = self.PITCH_COLOR
        
        self._draw_pitch(board)

        # Dibujar jugadores
        for pos, team_id in players:
            if team_id < 0:  # Saltar árbitro
                continue
            
            x = int(np.clip(pos[0], 5, self.width - 5))
            y = int(np.clip(pos[1], 5, self.height - 5))
            
            color = self.TEAM_COLORS[team_id % len(self.TEAM_COLORS)]
            
            cv2.circle(board, (x, y), 12, color, -1)
            cv2.circle(board, (x, y), 12, (255, 255, 255), 2)

        # Dibujar pelota
        if ball_mapped is not None:
            bx = int(np.clip(ball_mapped[0], 3, self.width - 3))
            by = int(np.clip(ball_mapped[1], 3, self.height - 3))
            cv2.circle(board, (bx, by), 7, (0, 165, 255), -1)
            cv2.circle(board, (bx, by), 8, (255, 255, 255), 1)

        return board


# ===================== SIMPLE FIELD MAPPER =====================

class SimpleFieldMapper:
    """Mapeo de coordenadas de cámara a tablero usando 6 puntos calibrados."""
    
    def __init__(self, field_rect: np.ndarray, board_width: int, board_height: int):
        """Inicializar mapeador con 6 puntos calibrados."""
        if len(field_rect) < 6:
            logger.warning(f"Se esperaban 6 puntos, se recibieron {len(field_rect)}")
            corners = np.array([
                field_rect[0],
                field_rect[2] if len(field_rect) > 2 else [field_rect[1][0] * 2 - field_rect[0][0], field_rect[0][1]],
                field_rect[3] if len(field_rect) > 3 else [field_rect[2][0], field_rect[1][1] * 2 - field_rect[0][1]],
                field_rect[5] if len(field_rect) > 5 else [field_rect[0][0], field_rect[3][1]],
            ], dtype=np.float32)
        else:
            corners = np.array([
                field_rect[0],  # TL
                field_rect[2],  # TR
                field_rect[3],  # BR
                field_rect[5],  # BL
            ], dtype=np.float32)
        
        dst_pts = np.array([
            [0, 0],
            [board_width, 0],
            [board_width, board_height],
            [0, board_height]
        ], dtype=np.float32)
        
        self.M = cv2.getPerspectiveTransform(corners, dst_pts)
        self.board_width = board_width
        self.board_height = board_height
        
        logger.info(f"Mapeador de campo inicializado con {len(field_rect)} puntos")
    
    def transform(self, pt: np.ndarray) -> np.ndarray:
        """Transformar punto de espacio de cámara a espacio de tablero."""
        if len(pt) < 2 or pt[0] < 0 or pt[1] < 0:
            return np.array([0.0, 0.0])
        
        try:
            pt_2d = pt.reshape(1, 1, 2).astype(np.float32)
            transformed = cv2.perspectiveTransform(pt_2d, self.M)[0][0]
            return transformed
        except Exception as e:
            logger.error(f"Error en transformación: {e}")
            return np.array([0.0, 0.0])


# ===================== PROCESS FRAME =====================

def process_frame(frame: np.ndarray,
                  model,
                  classifier: TeamClassifier,
                  mapper: SimpleFieldMapper,
                  field_validator: FieldValidator,
                  board_drawer: TacticalBoard,
                  ball_tracker: SimpleBallTracker,
                  config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Procesar un frame: detectar, clasificar y renderizar."""
    import supervision as sv
    
    # Detección YOLO
    results = model.predict(
        frame,
        classes=[config.player_class_id, config.ball_class_id],
        conf=config.yolo_conf_threshold,
        verbose=False,
        iou=0.5,
    )[0]
    
    detections = sv.Detections.from_ultralytics(results)
    logger.info(f"[DETECT] Total detecciones: {len(detections)}")

    player_dets = detections[detections.class_id == config.player_class_id]
    ball_dets = detections[detections.class_id == config.ball_class_id]
    
    logger.info(f"[DETECT] Jugadores: {len(player_dets)} | Balón: {len(ball_dets)}")

    # Filtrar jugadores por tamaño
    if len(player_dets) > 0:
        bbox_areas = (player_dets.xyxy[:, 2] - player_dets.xyxy[:, 0]) * \
                     (player_dets.xyxy[:, 3] - player_dets.xyxy[:, 1])
        min_area = 300
        size_mask = bbox_areas > min_area
        player_dets = player_dets[size_mask]

    # Filtrar jugadores fuera del campo
    if len(player_dets) > 0:
        feet_coords = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        valid_mask = field_validator.filter_detections(player_dets, feet_coords)
        player_dets = player_dets[valid_mask]

    # Entrenar clasificador
    if not classifier.trained and len(player_dets) >= config.min_players_for_kmeans:
        classifier.train_teams(frame, player_dets, config)
    
    # Mapear posiciones de jugadores al tablero
    players_frame: List[Tuple[np.ndarray, int]] = []
    if len(player_dets) > 0:
        feet = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        for i, foot in enumerate(feet):
            team_id = classifier.predict_team(frame, player_dets.xyxy[i])
            mapped = mapper.transform(foot)
            players_frame.append((mapped, team_id))

    # Procesar balón - extraer todas las detecciones y pasarlas al tracker
    ball_detections = []
    if len(ball_dets) > 0:
        for i in range(len(ball_dets)):
            ball_center = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[i]
            if field_validator.is_within_field(ball_center):
                ball_mapped = mapper.transform(ball_center)
                ball_detections.append(ball_mapped)
    
    mapped_ball = ball_tracker.update(ball_detections if ball_detections else None)

    # Anotar frame de cámara
    annotated = frame.copy()
    for i, bbox in enumerate(player_dets.xyxy):
        team_id = classifier.predict_team(frame, bbox)
        color = (128, 128, 128) if team_id < 0 else TacticalBoard.TEAM_COLORS[team_id % 2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    
    # Dibujar balón en cámara
    if len(ball_dets) > 0:
        for i in range(len(ball_dets)):
            ball_center = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[i]
            bx, by = int(ball_center[0]), int(ball_center[1])
            cv2.circle(annotated, (bx, by), 8, (0, 165, 255), -1)
            cv2.circle(annotated, (bx, by), 9, (255, 255, 255), 2)

    # Renderizar tablero
    board_img = board_drawer.draw_state(players_frame, mapped_ball)

    return annotated, board_img


# ===================== HELPERS =====================

def parse_start_time(time_str: str) -> int:
    """Parsear string de tiempo MM:SS o SS."""
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        else:
            return int(parts[0])
    except (ValueError, IndexError):
        logger.warning(f"Formato de tiempo inválido '{time_str}' - comenzando desde 0")
        return 0


def read_first_frame(cap: cv2.VideoCapture, config: Config) -> Optional[np.ndarray]:
    """Leer primer frame del video."""
    for attempt in range(config.stream_read_retries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            logger.info(f"Primer frame obtenido después de {attempt} intentos")
            return frame
        
        if attempt < config.stream_read_retries - 1:
            time.sleep(config.stream_read_delay)
    
    logger.error("No se pudo leer el primer frame")
    return None


def setup_detectors(config: Config) -> Optional['YOLO']:
    """Inicializar modelo YOLO."""
    logger.info("[SETUP] Cargando YOLO...")
    try:
        from ultralytics import YOLO
        logger.info("[OK] Importación exitosa")
    except ImportError as e:
        logger.error(f"[ERROR] Fallo importando: {e}")
        return None
    
    try:
        logger.info(f"[YOLO] Cargando modelo '{config.model_name}'...")
        model = YOLO(config.model_name)
        logger.info("[OK] Modelo YOLO cargado")
    except Exception as e:
        logger.error(f"[ERROR] Fallo cargando modelo: {e}")
        return None
    
    return model


# ===================== MAIN =====================

def main(cfg: Optional[Config] = None) -> None:
    """Pipeline principal de análisis."""
    cfg = cfg or Config()
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + "FUTSAL ANALYTICS - REAL-TIME TEAM ANALYSIS".center(68) + "║")
    print("║" + "Análisis de Futsal con Detección de Equipos".center(68) + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    logger.info("["*10 + "STARTUP" + "]"*10)
    
    # Get URL from user
    print("📺 YOUTUBE VIDEO URL")
    print("-" * 70)
    url = input("Enter YouTube URL (or 'q' to quit): ").strip()
    
    if url.lower() == 'q' or not url:
        logger.warning("User cancelled operation")
        return
    
    # Get optional start time
    print("\n⏱️  START TIME (Optional)")
    print("-" * 70)
    print("Format: MM:SS (e.g., 02:30)")
    time_str = input("Start time: ").strip()
    
    # Abrir stream
    logger.info("[STREAM] Abriendo stream de YouTube...")
    cap = open_youtube_stream(url, cfg)
    if cap is None or not cap.isOpened():
        logger.error("[ERROR] Falló al abrir stream")
        return
    
    logger.info("[OK] Stream abierto exitosamente")

    # Buscar minuto de inicio
    start_seconds = parse_start_time(time_str)
    if start_seconds > 0:
        logger.info(f"[SEEK] Buscando frame en {start_seconds}s...")
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info(f"[VIDEO] FPS: {fps}")

    # Leer primer frame
    first_frame = read_first_frame(cap, cfg)
    if first_frame is None:
        logger.error("[ERROR] No se pudo leer el primer frame")
        cap.release()
        return
    
    frame_h, frame_w = first_frame.shape[:2]
    logger.info(f"[FRAME] Dimensiones: {frame_w}x{frame_h}")

    # Calibración
    logger.info("[CALIBRATION] Abriendo interfaz de calibración...")
    calibrator = FieldCalibrator(first_frame.copy())
    field_rect = calibrator.calibrate()
    
    if len(field_rect) < 4:
        logger.error(f"[ERROR] Calibración falló")
        cap.release()
        return
    
    logger.info(f"[CALIBRATION] Calibración completada con {len(field_rect)} puntos")

    # Inicializar componentes
    logger.info("[INIT] Inicializando componentes...")
    try:
        field_validator = FieldValidator(field_rect)
        mapper = SimpleFieldMapper(field_rect, cfg.board_width, cfg.board_height)
    except Exception as e:
        logger.error(f"[ERROR] Error en setup: {e}")
        cap.release()
        return

    board_drawer = TacticalBoard(cfg.board_width, cfg.board_height)
    classifier = TeamClassifier()
    ball_tracker = SimpleBallTracker(max_speed=200.0, smoothing_alpha=0.5)

    logger.info("Inicializando YOLO...")
    model = setup_detectors(cfg)
    if model is None:
        logger.error("FALLÓ: No se pudo cargar YOLO")
        cap.release()
        return

    # Crear ventanas
    cv2.namedWindow("Vista Cámara", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tablero Táctico", cv2.WINDOW_NORMAL)

    frame_idx = 0
    t_start = time.time()
    logger.info("="*70)
    logger.info("ANÁLISIS EN TIEMPO REAL INICIADO - Presiona Q para detener")
    logger.info("="*70 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("[STREAM] Fin del stream alcanzado")
                break

            frame_idx += 1
            
            if frame_idx % 10 == 0:
                logger.info(f"[PROGRESS] Frame {frame_idx} - {time.time() - t_start:.1f}s")

            try:
                annotated, board_img = process_frame(
                    frame, model, classifier, mapper,
                    field_validator, board_drawer, ball_tracker, cfg
                )
            except Exception as e:
                logger.error(f"[ERROR] Frame {frame_idx}: {e}")
                continue

            # Overlay
            elapsed = time.time() - t_start
            status_text = f"Frame {frame_idx} | {elapsed:.1f}s"
            cv2.putText(annotated, status_text,
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Vista Cámara", annotated)
            cv2.imshow("Tablero Táctico", board_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("[USER] Análisis detenido por usuario")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - t_start
        fps_avg = frame_idx / elapsed if elapsed > 0 else 0
        
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + "✅ ANÁLISIS COMPLETADO".center(68) + "║")
        print("║" + f"Frames Procesados: {frame_idx}".ljust(68) + "║")
        print("║" + f"Tiempo Total: {elapsed:.1f}s".ljust(68) + "║")
        print("║" + f"FPS Promedio: {fps_avg:.1f}".ljust(68) + "║")
        print("╚" + "="*68 + "╝")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario")
    except Exception as e:
        logger.exception(f"Error inesperado: {e}")
