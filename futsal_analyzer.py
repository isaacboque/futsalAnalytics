"""
Futsal Team Analysis System
Análisis de estadísticas de equipos de futsal mediante cámara fija lateral.
Detección y clasificación de equipos con validación de jugadores dentro del campo.
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import csv
import time
import os
import subprocess
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# ===================== LOGGING SETUP =====================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ===================== CONFIGURATION CLASS =====================

@dataclass
class Config:
    """Configuración centralizada del sistema de análisis."""
    model_name: str = "yolo11n.pt"
    board_width: int = 1000
    board_height: int = 500
    max_display_width: int = 1280
    
    player_class_id: int = 0
    ball_class_id: int = 32
    
    min_players_for_train: int = 8
    min_players_for_kmeans: int = 6
    
    smoothing_window: int = 5
    duel_radius_px: int = 40
    possession_radius_px: int = 50
    
    yolo_conf_threshold: float = 0.3
    track_activation_threshold: float = 0.25
    
    output_csv: str = "analisis_futsal.csv"
    stream_read_retries: int = 30
    stream_read_delay: float = 0.3
    yt_dlp_timeout: int = 30
    
    def __post_init__(self):
        """Validar valores de configuración."""
        if self.board_width <= 0 or self.board_height <= 0:
            raise ValueError("Dimensiones del tablero deben ser positivas")
        if self.smoothing_window < 1:
            raise ValueError("Ventana de suavizado debe ser >= 1")


config = Config()


# ===================== YOUTUBE STREAM =====================

def open_youtube_stream(url: str, config: Config) -> Optional[cv2.VideoCapture]:
    """
    Abre un video de YouTube como OpenCV VideoCapture.
    Intenta obtener la mejor calidad disponible.
    
    Args:
        url: URL del video de YouTube
        config: Objeto de configuración
        
    Returns:
        cv2.VideoCapture si es exitoso, None en caso contrario
    """
    formats = ["best", "best[height<=1080]", "best[height<=720]", "best[height<=480]", "best[height<=360]"]
    
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


# ===================== FIELD CALIBRATION =====================

class FieldCalibrator:
    """
    Sistema de calibración por rectángulo ajustable para definir los límites del campo.
    Permite dibujar y ajustar un rectángulo que representa el área del futsal.
    NO RECORTA LA IMAGEN - Trabaja con resolución original.
    """
    
    def __init__(self, frame: np.ndarray, config: Config):
        """Inicializa el calibrador sin modificar la imagen."""
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        
        # Inicializar rectángulo por defecto (área central del frame)
        margin = max(100, int(min(self.w, self.h) * 0.1))
        self.rect = {
            'x1': margin,
            'y1': margin,
            'x2': self.w - margin,
            'y2': self.h - margin
        }
        
        self.dragging = False
        self.handle = None
        self.WIN = "CALIBRACIÓN DE CAMPO - Ajusta el Rectángulo"
    
    def normalize_rect(self):
        """Asegurar que x1 < x2 y y1 < y2."""
        if self.rect['x1'] > self.rect['x2']:
            self.rect['x1'], self.rect['x2'] = self.rect['x2'], self.rect['x1']
        if self.rect['y1'] > self.rect['y2']:
            self.rect['y1'], self.rect['y2'] = self.rect['y2'], self.rect['y1']
        
        # Limitaciones de límites
        self.rect['x1'] = max(0, min(self.rect['x1'], self.w - 1))
        self.rect['x2'] = max(0, min(self.rect['x2'], self.w - 1))
        self.rect['y1'] = max(0, min(self.rect['y1'], self.h - 1))
        self.rect['y2'] = max(0, min(self.rect['y2'], self.h - 1))
    
    def draw_overlay(self) -> np.ndarray:
        """Dibuja el rectángulo y las instrucciones sin recortar."""
        img = self.frame.copy()
        
        # Oscurecer área fuera del rectángulo
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), -1)
        cv2.rectangle(overlay, 
                     (self.rect['x1'], self.rect['y1']),
                     (self.rect['x2'], self.rect['y2']),
                     (0, 100, 0), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # Dibujar rectángulo principal - línea verde gruesa
        cv2.rectangle(img,
                     (self.rect['x1'], self.rect['y1']),
                     (self.rect['x2'], self.rect['y2']),
                     (0, 255, 0), 4)
        
        # Dibujar asas de ajuste - esquinas interactivas
        handle_size = 15
        handles = [
            (self.rect['x1'], self.rect['y1']),  # arriba-izquierda
            (self.rect['x2'], self.rect['y1']),  # arriba-derecha
            (self.rect['x2'], self.rect['y2']),  # abajo-derecha
            (self.rect['x1'], self.rect['y2']),  # abajo-izquierda
        ]
        for x, y in handles:
            cv2.circle(img, (x, y), handle_size, (0, 255, 255), -1)
            cv2.circle(img, (x, y), handle_size + 2, (255, 255, 255), 2)
        
        # Superposición de instrucciones
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # Texto
        cv2.putText(img, "CALIBRACIÓN DE CAMPO", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(img, "Arrastra las esquinas (puntos amarillos) para ajustar el rectángulo", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(img, "ESPACIO: Confirmar  |  R: Resetear  |  ESC: Cancelar", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        return img
    
    def get_handle_at(self, x: int, y: int) -> Optional[str]:
        """Obtener qué asa está siendo arrastrada."""
        threshold = 20
        handles = {
            'tl': (self.rect['x1'], self.rect['y1']),
            'tr': (self.rect['x2'], self.rect['y1']),
            'br': (self.rect['x2'], self.rect['y2']),
            'bl': (self.rect['x1'], self.rect['y2']),
        }
        for name, (hx, hy) in handles.items():
            if abs(x - hx) < threshold and abs(y - hy) < threshold:
                return name
        return None
    
    def on_mouse(self, event, x, y, flags, param):
        """Callback del mouse completamente reactivo."""
        if event == cv2.EVENT_LBUTTONDOWN:
            handle = self.get_handle_at(x, y)
            if handle:
                self.dragging = True
                self.handle = handle
                logger.debug(f"Comenzado arrastrar {handle}")
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.handle:
                if self.handle == 'tl':
                    self.rect['x1'], self.rect['y1'] = x, y
                elif self.handle == 'tr':
                    self.rect['x2'], self.rect['y1'] = x, y
                elif self.handle == 'br':
                    self.rect['x2'], self.rect['y2'] = x, y
                elif self.handle == 'bl':
                    self.rect['x1'], self.rect['y2'] = x, y
                self.normalize_rect()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                logger.debug(f"Finalizado arrastrar {self.handle}")
            self.dragging = False
            self.handle = None
    
    def calibrate(self) -> np.ndarray:
        """Ejecutar la interfaz de calibración completamente reactiva."""
        logger.info("="*70)
        logger.info(" "*10 + "CALIBRACIÓN DE CAMPO - FUTSAL")
        logger.info("="*70)
        logger.info("Instrucciones:")
        logger.info("  1. Los puntos amarillos en las esquinas son ARRASTABLES")
        logger.info("  2. Arrastra cada esquina hasta los límites reales del campo")
        logger.info("  3. Puedes mover el rectángulo fuera de la imagen si lo necesitas")
        logger.info("  4. Presiona ESPACIO cuando hayas terminado")
        logger.info("  5. Presiona R para resetear a los valores por defecto")
        logger.info("="*70 + "\n")
        
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WIN, self.on_mouse)
        
        while True:
            img = self.draw_overlay()
            cv2.imshow(self.WIN, img)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):  # ESPACIO
                logger.info("✓ Calibración confirmada\n")
                break
            elif key == ord('r') or key == ord('R'):  # R para resetear
                margin = max(100, int(min(self.w, self.h) * 0.1))
                self.rect = {
                    'x1': margin,
                    'y1': margin,
                    'x2': self.w - margin,
                    'y2': self.h - margin
                }
                logger.info("Rectángulo reseteado a valores por defecto")
            elif key == 27:  # ESC
                logger.warning("Calibración cancelada")
                break
        
        cv2.destroyAllWindows()
        
        # Retornar los puntos en formato array
        return np.array([
            [self.rect['x1'], self.rect['y1']],
            [self.rect['x2'], self.rect['y1']],
            [self.rect['x2'], self.rect['y2']],
            [self.rect['x1'], self.rect['y2']],
        ], dtype=np.float32)


# ===================== FIELD VALIDATOR =====================

class FieldValidator:
    """
    Valida si los jugadores detectados están dentro de los límites del campo.
    """

    def __init__(self, field_rect: np.ndarray):
        """
        Inicializar validador de campo.
        
        Args:
            field_rect: Array de 4 puntos que definen el rectángulo del campo
        """
        if len(field_rect) != 4:
            raise ValueError("Se necesitan exactamente 4 puntos")
        
        # Crear polígono convexo del rectángulo
        self.field_polygon = cv2.convexHull(field_rect.astype(np.float32))
        self.x_min = float(np.min(field_rect[:, 0]))
        self.x_max = float(np.max(field_rect[:, 0]))
        self.y_min = float(np.min(field_rect[:, 1]))
        self.y_max = float(np.max(field_rect[:, 1]))
        
        logger.info(f"Validador de campo inicializado: [{self.x_min:.0f}, {self.y_min:.0f}] a [{self.x_max:.0f}, {self.y_max:.0f}]")

    def is_within_field(self, point: np.ndarray) -> bool:
        """
        Verificar si un punto está dentro del campo.
        
        Args:
            point: Punto [x, y]
            
        Returns:
            True si el punto está dentro del campo
        """
        # Verificación rápida de límites primero
        if point[0] < self.x_min or point[0] > self.x_max:
            return False
        if point[1] < self.y_min or point[1] > self.y_max:
            return False
        
        # Verificación poligonal
        result = cv2.pointPolygonTest(self.field_polygon, point[:2], False)
        return result >= 0

    def filter_detections(self, detections: sv.Detections, feet_coords: np.ndarray) -> np.ndarray:
        """
        Filtrar detecciones para mantener solo las dentro del campo.
        
        Args:
            detections: Detecciones de supervision
            feet_coords: Coordenadas de pies para cada detección
            
        Returns:
            Máscara booleana de detecciones válidas
        """
        valid_mask = np.array([self.is_within_field(foot) for foot in feet_coords])
        return valid_mask


# ===================== TEAM CLASSIFIER =====================

class TeamClassifier:
    """
    Separa jugadores en dos equipos usando K-Means en color de camiseta (HSV).
    """

    def __init__(self, n_clusters: int = 3, n_init: int = 20):
        """Inicializar el clasificador."""
        self.kmeans: KMeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
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
        """
        Extraer características de color HSV de la camiseta.
        
        Args:
            frame: Frame de entrada
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Vector de características [hue, saturation, value]
        """
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

    def train_teams(self, frame: np.ndarray, detections: sv.Detections, config: Config) -> bool:
        """
        Entrenar clasificador K-Means en colores de camiseta.
        
        Args:
            frame: Frame de entrada
            detections: Detecciones
            config: Configuración
            
        Returns:
            True si fue exitoso
        """
        if len(detections) < config.min_players_for_kmeans:
            logger.debug(f"No hay suficientes jugadores ({len(detections)}) para entrenar")
            return False
        
        try:
            color_features = np.array(
                [self.get_jersey_color_features(frame, b) for b in detections.xyxy],
                dtype=np.float32
            )
            
            self.color_features = color_features
            hue_var = np.var(color_features[:, 0])
            sat_var = np.var(color_features[:, 1])
            
            if hue_var < 50:
                logger.warning("Varianza de color insuficiente para clasificación de equipos")
                return False
            
            self.kmeans.fit(color_features)
            self.trained = True
            
            counts = np.bincount(self.kmeans.labels_)
            self.ref_label = int(np.argmin(counts))
            
            logger.info(
                f"Clasificador de equipos entrenado exitosamente. "
                f"Árbitro: cluster {self.ref_label} | "
                f"Tamaños: {counts}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error al entrenar equipos: {e}")
            return False

    def predict_team(self, frame: np.ndarray, bbox: np.ndarray) -> int:
        """
        Predecir equipo para un jugador.
        
        Args:
            frame: Frame de entrada
            bbox: Bounding box del jugador
            
        Returns:
            0 o 1 para equipos, -1 para árbitro
        """
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


# ===================== POSITION SMOOTHER =====================

class PositionSmoother:
    """Suavizado de posición mediante promedio rodante."""

    def __init__(self, window: int = 5):
        """Inicializar con tamaño de ventana."""
        self.buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, pos: np.ndarray) -> np.ndarray:
        """
        Actualizar posición e retornar posición suavizada.
        
        Args:
            track_id: ID del jugador
            pos: Vector de posición [x, y]
            
        Returns:
            Posición suavizada
        """
        self.buffers[track_id].append(pos)
        return np.mean(self.buffers[track_id], axis=0)


# ===================== BALL TRACKER =====================

class BallTracker:
    """
    Rastreador de balón simple usando centroide más cercano frame a frame.
    """
    
    def __init__(self, max_distance: float = 100):
        """
        Inicializar rastreador de balón.
        
        Args:
            max_distance: Distancia máxima entre frames para considerar mismo balón
        """
        self.last_pos: Optional[np.ndarray] = None
        self.max_distance = max_distance
        self.ball_seen_count = 0
        self.ball_positions: List[np.ndarray] = []
    
    def update(self, current_pos: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Actualizar posición del balón con suavizado temporal.
        
        Args:
            current_pos: Posición actual del balón o None
            
        Returns:
            Posición rastreada o None
        """
        if current_pos is None:
            self.last_pos = None
            return None
        
        # Validar continuidad del balón
        if self.last_pos is not None:
            distance = np.linalg.norm(current_pos - self.last_pos)
            if distance > self.max_distance:
                logger.debug(f"Posible desconexión de balón (distancia: {distance:.1f})")
                self.ball_positions = []
        
        self.last_pos = current_pos.copy()
        self.ball_seen_count += 1
        
        # Mantener historial de posiciones
        self.ball_positions.append(current_pos.copy())
        if len(self.ball_positions) > 5:
            self.ball_positions = self.ball_positions[-5:]
        
        # Retornar promedio de últimas posiciones (suavizado)
        return np.mean(self.ball_positions, axis=0)


# ===================== TACTICAL BOARD =====================

class TacticalBoard:
    """
    Renderiza un diagrama overhead del campo de futsal con jugadores y pelota.
    """

    TEAM_COLORS: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((255, 80, 80), (80, 255, 255))
    PITCH_COLOR: Tuple[int, int, int] = (34, 139, 34)

    def __init__(self, width: int, height: int):
        """Inicializar tablero táctico."""
        self.width: int = width
        self.height: int = height

    def _draw_pitch(self, board: np.ndarray) -> None:
        """Dibujar líneas del campo."""
        line_color = (255, 255, 255)
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Límites del campo
        cv2.rectangle(board, (0, 0), (self.width - 1, self.height - 1), line_color, 3)
        
        # Línea de centro
        cv2.line(board, (center_x, 0), (center_x, self.height), line_color, 2)
        
        # Punto central
        cv2.circle(board, (center_x, center_y), 5, line_color, -1)
        cv2.circle(board, (center_x, center_y), 60, line_color, 2)
        
        # Áreas de penales
        pen_box_w, pen_box_h = 120, 200
        goal_box_w, goal_box_h = 60, 100
        
        # Lado izquierdo
        left_pen_y1 = center_y - pen_box_h // 2
        left_pen_y2 = center_y + pen_box_h // 2
        left_goal_y1 = center_y - goal_box_h // 2
        left_goal_y2 = center_y + goal_box_h // 2
        
        cv2.rectangle(board, (0, left_pen_y1), (pen_box_w, left_pen_y2), line_color, 2)
        cv2.rectangle(board, (0, left_goal_y1), (goal_box_w, left_goal_y2), line_color, 2)
        cv2.circle(board, (int(pen_box_w * 0.6), center_y), 3, line_color, -1)
        
        # Lado derecho
        right_pen_y1 = center_y - pen_box_h // 2
        right_pen_y2 = center_y + pen_box_h // 2
        right_goal_y1 = center_y - goal_box_h // 2
        right_goal_y2 = center_y + goal_box_h // 2
        
        cv2.rectangle(board, (self.width - pen_box_w, right_pen_y1), 
                      (self.width, right_pen_y2), line_color, 2)
        cv2.rectangle(board, (self.width - goal_box_w, right_goal_y1), 
                      (self.width, right_goal_y2), line_color, 2)
        cv2.circle(board, (self.width - int(pen_box_w * 0.6), center_y), 3, line_color, -1)

    def draw_state(self, players: List[Tuple[int, np.ndarray, int]], 
                   ball_mapped: Optional[np.ndarray]) -> np.ndarray:
        """
        Dibujar estado actual del partido.
        
        Args:
            players: Lista de (track_id, position, team_id)
            ball_mapped: Posición de la pelota o None
            
        Returns:
            Imagen del tablero renderizada
        """
        board = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        board[:] = self.PITCH_COLOR
        
        self._draw_pitch(board)

        # Dibujar jugadores
        for track_id, pos, team_id in players:
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


# ===================== SIMPLE PERSPECTIVE MAPPER =====================

class SimpleFieldMapper:
    """
    Mapeo simple de coordenadas de cámara a coordenadas del tablero
    usando el rectángulo calibrado del campo.
    """
    
    def __init__(self, field_rect: np.ndarray, board_width: int, board_height: int):
        """
        Inicializar mapeador.
        
        Args:
            field_rect: Array con 4 puntos del rectángulo calibrado
            board_width: Ancho del tablero
            board_height: Alto del tablero
        """
        # Crear matriz de perspectiva simple
        src_pts = field_rect.astype(np.float32)
        dst_pts = np.array([
            [0, 0],
            [board_width, 0],
            [board_width, board_height],
            [0, board_height]
        ], dtype=np.float32)
        
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.board_width = board_width
        self.board_height = board_height
    
    def transform(self, pt: np.ndarray) -> np.ndarray:
        """
        Transformar punto de espacio de cámara a espacio de tablero.
        
        Args:
            pt: Punto [x, y]
            
        Returns:
            Punto transformado [x, y]
        """
        if pt[0] < 0 or pt[1] < 0:
            return pt
        
        pt_2d = pt.reshape(1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pt_2d, self.M)[0][0]
        
        return transformed


# ===================== HELPER FUNCTIONS =====================

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
            logger.debug(f"Esperando buffer de stream... ({attempt + 1}/{config.stream_read_retries})")
            time.sleep(config.stream_read_delay)
    
    logger.error("No se pudo leer el primer frame")
    return None


def setup_detectors(config: Config) -> Tuple[Optional[YOLO], Optional[sv.ByteTrack]]:
    """Inicializar modelo YOLO y tracker ByteTrack."""
    try:
        model = YOLO(config.model_name)
        logger.info(f"Modelo YOLO '{config.model_name}' cargado exitosamente")
    except Exception as e:
        logger.error(f"Error cargando modelo YOLO: {e}")
        return None, None
    
    tracker = sv.ByteTrack(track_activation_threshold=config.track_activation_threshold)
    logger.info("ByteTrack tracker inicializado")
    
    return model, tracker


def process_frame(frame: np.ndarray,
                  model: YOLO,
                  tracker: sv.ByteTrack,
                  classifier: TeamClassifier,
                  smoother: PositionSmoother,
                  mapper: SimpleFieldMapper,
                  field_validator: FieldValidator,
                  board_drawer: TacticalBoard,
                  ball_tracker: BallTracker,
                  config: Config,
                  fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procesar un frame: detectar, rastrear, clasificar, validar y renderizar.
    Incluye rastreo de balón.
    """
    # Detección YOLO
    results = model.predict(
        frame,
        classes=[config.player_class_id, config.ball_class_id],
        conf=config.yolo_conf_threshold,
        verbose=False
    )[0]
    detections = sv.Detections.from_ultralytics(results)

    player_dets = detections[detections.class_id == config.player_class_id]
    ball_dets = detections[detections.class_id == config.ball_class_id]

    # Filtrar jugadores fuera del campo
    if len(player_dets) > 0:
        feet_coords = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        valid_mask = field_validator.filter_detections(player_dets, feet_coords)
        player_dets = player_dets[valid_mask]
        logger.debug(f"Filtrados {np.sum(~valid_mask)} jugadores fuera del campo")

    # ByteTrack para IDs persistentes
    tracked = tracker.update_with_detections(player_dets)

    # Entrenar clasificador cuando haya suficientes jugadores
    if not classifier.trained and len(tracked) >= config.min_players_for_train:
        classifier.train_teams(frame, tracked, config)

    # Mapear posiciones de jugadores al tablero
    players_frame: List[Tuple[int, np.ndarray, int]] = []
    if len(tracked) > 0:
        feet = tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        for i, foot in enumerate(feet):
            tid = int(tracked.tracker_id[i])
            team_id = classifier.predict_team(frame, tracked.xyxy[i])
            mapped = mapper.transform(foot)
            smooth = smoother.update(tid, mapped)
            players_frame.append((tid, smooth, team_id))

    # Mapear y rastrear posición de la pelota
    current_ball_pos: Optional[np.ndarray] = None
    if len(ball_dets) > 0:
        ball_center = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
        if field_validator.is_within_field(ball_center):
            current_ball_pos = mapper.transform(ball_center)
    
    # Actualizar rastreador de balón
    mapped_ball = ball_tracker.update(current_ball_pos)

    # Anotar frame de cámara
    annotated = frame.copy()
    for i, bbox in enumerate(tracked.xyxy):
        team_id = classifier.predict_team(frame, bbox)
        color = (128, 128, 128) if team_id < 0 else TacticalBoard.TEAM_COLORS[team_id % 2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        team_text = "ÁRBITRO" if team_id < 0 else f"EQUIPO {team_id}"
        cv2.putText(annotated, team_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Dibujar balón rastreado
    if len(ball_dets) > 0:
        ball_center = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
        bx, by = int(ball_center[0]), int(ball_center[1])
        cv2.circle(annotated, (bx, by), 8, (0, 165, 255), -1)
        cv2.circle(annotated, (bx, by), 9, (255, 255, 255), 2)

    # Renderizar tablero táctico
    board_img = board_drawer.draw_state(players_frame, mapped_ball)

    return annotated, board_img


# ===================== MAIN =====================

def main(cfg: Optional[Config] = None) -> None:
    """Pipeline principal de análisis."""
    cfg = cfg or Config()
    
    # Obtener input del usuario
    print("\n" + "="*70)
    print(" "*15 + "ANALIZADOR DE FUTSAL - CÁMARA LATERAL")
    print("="*70 + "\n")
    
    url = input("Ingresa URL de YouTube: ").strip()
    if not url:
        logger.error("URL vacía")
        return

    start_time_str = input("Minuto de inicio (MM:SS, Enter para saltar): ").strip()
    
    # Abrir stream
    logger.info("Abriendo stream de YouTube...")
    cap = open_youtube_stream(url, cfg)
    if cap is None or not cap.isOpened():
        logger.error("Falló al abrir stream. Soluciones:")
        logger.error("  1. pip install --upgrade yt-dlp")
        logger.error("  2. Verifica que el video sea público")
        return

    # Buscar minuto de inicio
    start_seconds = parse_start_time(start_time_str)
    if start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)
        logger.info(f"Buscando a {start_seconds} segundos")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info(f"FPS del video: {fps}")

    # Leer primer frame
    first_frame = read_first_frame(cap, cfg)
    if first_frame is None:
        cap.release()
        return

    # Calibración con rectángulo
    calibrator = FieldCalibrator(first_frame.copy(), cfg)
    field_rect = calibrator.calibrate()
    
    if len(field_rect) != 4:
        logger.error(f"Calibración falló: se necesitaban 4 puntos, se obtuvieron {len(field_rect)}")
        cap.release()
        return

    # Inicializar componentes
    try:
        field_validator = FieldValidator(field_rect)
        mapper = SimpleFieldMapper(field_rect, cfg.board_width, cfg.board_height)
    except Exception as e:
        logger.error(f"Error en setup de validador: {e}")
        cap.release()
        return

    board_drawer = TacticalBoard(cfg.board_width, cfg.board_height)
    classifier = TeamClassifier()
    smoother = PositionSmoother(cfg.smoothing_window)
    ball_tracker = BallTracker(max_distance=150)

    model, tracker = setup_detectors(cfg)
    if model is None or tracker is None:
        cap.release()
        return

    # Crear ventanas
    cv2.namedWindow("Vista Cámara", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tablero Táctico", cv2.WINDOW_NORMAL)

    frame_idx = 0
    t_start = time.time()
    logger.info("Análisis iniciado - Presiona Q para detener\n")

    try:
        # Bucle principal
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("Fin del stream")
                break

            frame_idx += 1

            # Procesar frame
            try:
                annotated, board_img = process_frame(
                    frame, model, tracker, classifier, smoother, mapper,
                    field_validator, board_drawer, ball_tracker, cfg, fps
                )
            except Exception as e:
                logger.error(f"Error procesando frame {frame_idx}: {e}")
                continue

            # Overlay de contador
            elapsed = time.time() - t_start
            cv2.putText(annotated, f"Frame {frame_idx} | {elapsed:.0f}s",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Mostrar
            cv2.imshow("Vista Cámara", annotated)
            cv2.imshow("Tablero Táctico", board_img)

            # Verificar salida
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Análisis detenido por usuario")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.info(f"\nAnálisis completado - {frame_idx} frames procesados")
    logger.info("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario")
    except Exception as e:
        logger.exception(f"Error inesperado: {e}")
