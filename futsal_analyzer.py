"""
Futsal Team Analysis System
Análisis de estadísticas de equipos de futsal mediante cámara fija lateral.
Detección y clasificación de equipos con validación de jugadores dentro del campo.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
import time
import subprocess
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

# Imports pesados - se cargarán después de FieldCalibrator
# import supervision as sv
# from ultralytics import YOLO
# from sklearn.cluster import KMeans

# ===================== LOGGING SETUP =====================

logging.basicConfig(
    level=logging.DEBUG,
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
    
    player_class_id: int = 0
    ball_class_id: int = 32
    
    min_players_for_train: int = 8
    min_players_for_kmeans: int = 6
    
    smoothing_window: int = 5
    duel_radius_px: int = 40
    possession_radius_px: int = 50
    
    yolo_conf_threshold: float = 0.3
    track_activation_threshold: float = 0.25
    
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


class FieldCalibrator:
    """
    Sistema de calibración por 6 puntos independientes para definir los límites del campo.
    Permite ajustar libremente cada punto para adaptarse al ángulo de cámara.
    """
    
    def __init__(self, frame: np.ndarray):
        """Inicializa el calibrador con 6 puntos independientes."""
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        
        # Canvas con padding para mostrar el frame centrado
        self.padding = max(200, int(self.h * 0.3))  # 30% de padding o mínimo 200px
        self.canvas_w = self.w + 2 * self.padding
        self.canvas_h = self.h + 2 * self.padding
        self.offset_x = self.padding
        self.offset_y = self.padding
        
        # Inicializar 6 puntos como trapezio irregular: [TL, CT, TR, BR, CB, BL]
        # Puntos como array de coordenadas (x, y)
        margin_x = int(self.w * 0.1)
        margin_y = int(self.h * 0.15)
        cx = self.w // 2
        
        self.points = np.array([
            [margin_x, margin_y],                    # 0: TL (Top-Left)
            [cx, margin_y],                          # 1: CT (Central Top)
            [self.w - margin_x, margin_y],           # 2: TR (Top-Right)
            [self.w - margin_x, self.h - margin_y],  # 3: BR (Bottom-Right)
            [cx, self.h - margin_y],                 # 4: CB (Central Bottom)
            [margin_x, self.h - margin_y],           # 5: BL (Bottom-Left)
        ], dtype=np.float32)
        
        # Estados para el mouse
        self.dragging = False
        self.active_point = None
        self.WIN = "Calibración de Campo"
    
    def draw_frame(self) -> np.ndarray:
        """Dibuja el frame con el polígono y 6 puntos de control, centrado en un canvas."""
        # Crear canvas gris
        canvas = np.full((self.canvas_h, self.canvas_w, 3), (80, 80, 80), dtype=np.uint8)
        
        # Colocar el frame en el centro del canvas
        canvas[self.offset_y:self.offset_y + self.h, self.offset_x:self.offset_x + self.w] = self.frame.copy()
        
        # Convertir puntos a enteros para dibujar (sumar offset al canvas)
        pts = (self.points + np.array([self.offset_x, self.offset_y])).astype(np.int32)
        
        # Oscurecer área fuera del polígono
        overlay = canvas.copy()
        cv2.rectangle(overlay, (self.offset_x, self.offset_y), 
                     (self.offset_x + self.w, self.offset_y + self.h), (0, 0, 0), -1)
        cv2.fillPoly(overlay, [pts], (50, 100, 50))
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
        
        # Dibujar frame border
        cv2.rectangle(canvas, (self.offset_x, self.offset_y), 
                     (self.offset_x + self.w - 1, self.offset_y + self.h - 1), (200, 200, 200), 2)
        
        # Dibujar polígono con líneas
        for i in range(len(pts)):
            next_i = (i + 1) % len(pts)
            cv2.line(canvas, tuple(pts[i]), tuple(pts[next_i]), (0, 255, 0), 3)
        
        # Dibujar línea central (entre CL y CR)
        if len(pts) >= 6:
            cv2.line(canvas, tuple(pts[4]), tuple(pts[5]), (0, 200, 200), 2, cv2.LINE_AA)
        
        # Dibujar 6 puntos de control
        handle_size = 12
        thickness = 3
        labels = ["TL", "CT", "TR", "BR", "CB", "BL"]
        
        for idx, (pt, label) in enumerate(zip(pts, labels)):
            x, y = int(pt[0]), int(pt[1])
            
            # Color diferente para punto activo
            if self.active_point == idx and self.dragging:
                cv2.circle(canvas, (x, y), handle_size, (0, 165, 255), -1)  # Naranja
            else:
                cv2.circle(canvas, (x, y), handle_size, (0, 255, 255), -1)  # Amarillo
            
            cv2.circle(canvas, (x, y), handle_size, (255, 255, 255), thickness)  # Borde blanco
            
            # Dibujar número del punto
            cv2.putText(canvas, str(idx), (x-5, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Etiqueta descriptiva
            cv2.putText(canvas, label, (x-15, y-handle_size-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Información en pantalla
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
        """Actualiza la posición de un punto de control de forma completamente libre, incluso fuera del frame."""
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
        print("\nINSTRUCCIONES:")
        print("  • Ajusta cada punto a los límites reales del campo")
        print("  • Los puntos pueden estar fuera del frame")
        print("  • Espacio dentro del campo = área de juego válida")
        print("  • Esperado: ~4 jugadores por equipo + árbitro")
        print("\nCONTROLES:")
        print("  - CLIC + ARRASTRAR en puntos para ajustar")
        print("  - ESPACIO para confirmar")
        print("  - R para resetear")
        print("  - ESC para cancelar")
        print("="*70 + "\n")
        
        cv2.namedWindow(self.WIN)
        
        labels_map = {0: "TL", 1: "CT", 2: "TR", 3: "BR", 4: "CB", 5: "BL"}
        
        def mouse_event(event, x, y, flags, param):
            """Maneja eventos del ratón de forma robusta."""
            # Convertir coordenadas del canvas a coordenadas del frame
            frame_x = x - self.offset_x
            frame_y = y - self.offset_y
            
            if event == cv2.EVENT_LBUTTONDOWN:
                point_idx = self.get_point_index(frame_x, frame_y)
                if point_idx is not None:
                    self.dragging = True
                    self.active_point = point_idx
                    label = labels_map.get(point_idx, "?")
                    print(f"  -> PUNTO {label} ({point_idx}) ACTIVO - Arrastrando...")
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging and self.active_point is not None:
                    self.update_point(self.active_point, frame_x, frame_y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if self.dragging and self.active_point is not None:
                    label = labels_map.get(self.active_point, "?")
                    print(f"  [OK] Punto {label} ({self.active_point}) - LIBERADO")
                self.dragging = False
                self.active_point = None
        
        # Registrar el callback - CRÍTICO para la reactividad
        cv2.setMouseCallback(self.WIN, mouse_event)
        
        print("Ventana abierta. Mueve los 6 puntos amarillos.\n")
        
        while True:
            img = self.draw_frame()
            cv2.imshow(self.WIN, img)
            
            # Usar waitKey sin tiempo límite para mejor captura de eventos
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord(' '):
                print("[OK] Calibración confirmada\n")
                break
            elif key == ord('r') or key == ord('R'):
                # Resetear a posición por defecto
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
                print("[OK] Puntos reseteados\n")
            elif key == 27:  # ESC
                print("[CANCEL] Calibración cancelada\n")
                break
        
        cv2.destroyAllWindows()
        
        # Retornar el polígono del campo (6 puntos que definen las líneas del campo)
        # Estos puntos representan: TL, CT, TR, BR, CB, BL
        # y definen la geometría exacta del campo para validación y transformación
        return self.points.copy()


# ===================== FIELD VALIDATOR =====================

class FieldValidator:
    """
    Valida si los jugadores detectados están dentro de los límites del campo.
    """

    def __init__(self, field_rect: np.ndarray):
        """
        Inicializar validador de campo con el polígono exacto del campo.
        
        Args:
            field_rect: Array de 6 puntos que definen el polígono del campo
                       Puntos: TL, CT, TR, BR, CB, BL
                       que representan las líneas calibradas del campo
        """
        if len(field_rect) < 6:
            logger.warning(f"Se esperaban 6 puntos de calibración, se obtuvieron {len(field_rect)}")
            if len(field_rect) < 4:
                raise ValueError("Se necesitan al menos 4 puntos")
        
        # Usar todos los puntos para crear el polígono exacto del campo
        # Los 6 puntos definen las líneas del campo de forma precisa
        self.field_polygon = field_rect.astype(np.float32)
        
        # Calcular bounding box para optimización
        self.x_min = float(np.min(self.field_polygon[:, 0]))
        self.x_max = float(np.max(self.field_polygon[:, 0]))
        self.y_min = float(np.min(self.field_polygon[:, 1]))
        self.y_max = float(np.max(self.field_polygon[:, 1]))
        
        logger.info(f"Validador de campo inicializado con polígono de {len(self.field_polygon)} puntos")
        logger.debug(f"Bounding box: [{self.x_min:.0f}, {self.y_min:.0f}] a [{self.x_max:.0f}, {self.y_max:.0f}]")

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

    def filter_detections(self, detections, feet_coords: np.ndarray) -> np.ndarray:
        """
        Filtrar detecciones para mantener solo las dentro del campo.
        Validación robusta: verifica centroide Y bounding box.
        
        Args:
            detections: Detecciones de supervision
            feet_coords: Coordenadas de pies para cada detección
            
        Returns:
            Máscara booleana de detecciones válidas
        """
        valid_mask = []
        
        for i, foot in enumerate(feet_coords):
            # Validación con el polígono preciso del campo calibrado
            # La posición de los pies es el indicador principal
            foot_valid = self.is_within_field(foot)
            
            # Validación secundaria: centroide de bbox
            bbox = detections.xyxy[i]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            center = np.array([center_x, center_y])
            center_valid = self.is_within_field(center)
            
            # El jugador es válido si está dentro del polígono del campo
            # (al menos los pies deben estar cerca del perímetro del campo)
            is_valid = foot_valid or center_valid
            valid_mask.append(is_valid)
            
            if not is_valid:
                logger.debug(f"[FILTER] Jugador rechazado: foot={foot_valid}, center={center_valid}")
        
        return np.array(valid_mask)


# ===================== TEAM CLASSIFIER =====================

class TeamClassifier:
    """
    Separa jugadores en dos equipos usando K-Means en color de camiseta (HSV).
    """

    def __init__(self, n_clusters: int = 3, n_init: int = 20):
        """Inicializar el clasificador."""
        # Importar KMeans lazily
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

    def train_teams(self, frame: np.ndarray, detections, config: Config) -> bool:
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
            sat_var = np.var(color_features[:, 1])
            
            logger.debug(f"[TRAIN] Varianza Hue: {hue_var:.2f} | Varianza Sat: {sat_var:.2f}")
            
            if hue_var < 50:
                logger.warning("[TRAIN] Varianza de color insuficiente para clasificación")
                return False
            
            logger.info("[TRAIN] Entrenando K-Means...")
            self.kmeans.fit(color_features)
            self.trained = True
            
            counts = np.bincount(self.kmeans.labels_)
            self.ref_label = int(np.argmin(counts))
            
            logger.info(f"[TRAIN] K-Means entrenado exitosamente")
            logger.info(f"  - Árbitro detectado en cluster: {self.ref_label}")
            logger.info(f"  - Distribución de clusters: {list(counts)}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error entrenando equipos: {e}")
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
# Eliminado: PositionSmoother no es necesario sin tracking de IDs persistentes


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
    Mapeo de coordenadas de cámara a tablero usando 6 puntos calibrados.
    
    Los 6 puntos definen el polígono exacto del campo:
    - Índice 0: TL (Top-Left)
    - Índice 1: CT (Center-Top)
    - Índice 2: TR (Top-Right)
    - Índice 3: BR (Bottom-Right)
    - Índice 4: CB (Center-Bottom)
    - Índice 5: BL (Bottom-Left)
    """
    
    def __init__(self, field_rect: np.ndarray, board_width: int, board_height: int):
        """
        Inicializar mapeador con 6 puntos calibrados.
        
        Args:
            field_rect: Array de 6 puntos: [TL, CT, TR, BR, CB, BL]
            board_width: Ancho del tablero
            board_height: Alto del tablero
        """
        if len(field_rect) < 6:
            logger.warning(f"Se esperaban 6 puntos, se recibieron {len(field_rect)}. Usando 4 esquinas.")
            # Fallback: si faltan puntos, usar solo esquinas
            corners = np.array([
                field_rect[0],  # TL
                field_rect[2] if len(field_rect) > 2 else [field_rect[1][0] * 2 - field_rect[0][0], field_rect[0][1]],  # TR
                field_rect[3] if len(field_rect) > 3 else [field_rect[2][0], field_rect[1][1] * 2 - field_rect[0][1]],  # BR
                field_rect[5] if len(field_rect) > 5 else [field_rect[0][0], field_rect[3][1]],  # BL
            ], dtype=np.float32)
        else:
            # Usar los 4 puntos de esquina en el orden correcto: TL, TR, BR, BL
            corners = np.array([
                field_rect[0],  # TL (index 0)
                field_rect[2],  # TR (index 2)
                field_rect[3],  # BR (index 3)
                field_rect[5],  # BL (index 5)
            ], dtype=np.float32)
        
        # Los puntos del medio se usan para refinar la transformación
        self.field_rect = field_rect.astype(np.float32)
        self.ct = field_rect[1] if len(field_rect) > 1 else None  # Center-Top
        self.cb = field_rect[4] if len(field_rect) > 4 else None  # Center-Bottom
        
        # Destino: rectángulo completo del tablero
        dst_pts = np.array([
            [0, 0],                    # TL
            [board_width, 0],          # TR
            [board_width, board_height],  # BR
            [0, board_height]          # BL
        ], dtype=np.float32)
        
        # Matriz de transformación perspectiva usando 4 esquinas
        self.M = cv2.getPerspectiveTransform(corners, dst_pts)
        self.board_width = board_width
        self.board_height = board_height
        
        logger.info(f"Mapeador de campo inicializado con {len(field_rect)} puntos")
    
    def transform(self, pt: np.ndarray) -> np.ndarray:
        """
        Transformar punto de espacio de cámara a espacio de tablero.
        Usa la matriz de perspectiva calculada con los 4 puntos de esquina.
        
        Args:
            pt: Punto [x, y]
            
        Returns:
            Punto transformado [x, y] en coordenadas del tablero
        """
        if len(pt) < 2 or pt[0] < 0 or pt[1] < 0:
            logger.debug(f"Punto inválido: {pt}")
            return np.array([0.0, 0.0])
        
        try:
            pt_2d = pt.reshape(1, 1, 2).astype(np.float32)
            transformed = cv2.perspectiveTransform(pt_2d, self.M)[0][0]
            
            # Validar que el punto transformado esté dentro de los límites del tablero
            if transformed[0] < 0 or transformed[0] > self.board_width or \
               transformed[1] < 0 or transformed[1] > self.board_height:
                logger.debug(f"Punto transformado fuera de tablero: {transformed}")
            
            return transformed
        except Exception as e:
            logger.error(f"Error en transformación: {e}")
            return np.array([0.0, 0.0])


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


def setup_detectors(config: Config) -> Optional['YOLO']:
    """Inicializar modelo YOLO para detección en tiempo real."""
    # Cargar imports pesados solo cuando se necesiten
    logger.info("[SETUP] Cargando dependencias pesadas (ultralytics, sklearn)...")
    try:
        logger.debug("[IMPORT] from ultralytics import YOLO")
        from ultralytics import YOLO
        logger.info("[OK] Dependencias importadas correctamente")
    except ImportError as e:
        logger.error(f"[ERROR] Fallo importando dependencias: {e}")
        return None
    
    try:
        logger.info(f"[YOLO] Cargando modelo '{config.model_name}'...")
        model = YOLO(config.model_name)
        logger.info(f"[OK] Modelo YOLO cargado en memoria")
    except Exception as e:
        logger.error(f"[ERROR] Fallo cargando modelo YOLO: {e}")
        return None
    
    logger.info("[SETUP] Detector listo: YOLO")
    return model


def process_frame(frame: np.ndarray,
                  model,
                  classifier: TeamClassifier,
                  mapper: SimpleFieldMapper,
                  field_validator: FieldValidator,
                  board_drawer: TacticalBoard,
                  ball_tracker: BallTracker,
                  config: Config,
                  fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Procesar un frame: detectar, clasificar, validar y renderizar en tiempo real.
    Sin tracking de IDs persistentes.
    """
    import supervision as sv
    
    logger.debug("[PROCESS] Iniciando procesamiento de frame...")
    
    # Detección YOLO
    logger.debug("[YOLO] Ejecutando detección YOLO...")
    results = model.predict(
        frame,
        classes=[config.player_class_id, config.ball_class_id],
        conf=config.yolo_conf_threshold,
        verbose=False,
        iou=0.5,
    )[0]
    logger.debug(f"[YOLO] Detecciones encontradas: {len(results.boxes)}")
    
    detections = sv.Detections.from_ultralytics(results)
    logger.info(f"[DETECT] Total detecciones: {len(detections)}")

    player_dets = detections[detections.class_id == config.player_class_id]
    ball_dets = detections[detections.class_id == config.ball_class_id]
    
    logger.info(f"[DETECT] Jugadores brutos: {len(player_dets)} | Balón: {len(ball_dets)}")

    # Filtrar jugadores por tamaño
    if len(player_dets) > 0:
        bbox_areas = (player_dets.xyxy[:, 2] - player_dets.xyxy[:, 0]) * \
                     (player_dets.xyxy[:, 3] - player_dets.xyxy[:, 1])
        min_area = 300  # Reducido para detectar más jugadores
        size_mask = bbox_areas > min_area
        
        if np.sum(~size_mask) > 0:
            logger.info(f"[FILTER] Detectadas {np.sum(~size_mask)} detecciones muy pequeñas (ruido)")
        
        player_dets = player_dets[size_mask]

    # Filtrar jugadores fuera del campo
    filtered_count = 0
    if len(player_dets) > 0:
        feet_coords = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        valid_mask = field_validator.filter_detections(player_dets, feet_coords)
        filtered_count = np.sum(~valid_mask)
        player_dets = player_dets[valid_mask]
        logger.info(f"[FILTER] Jugadores fuera del campo: {filtered_count} | Válidos dentro: {len(player_dets)}")

    # Entrenar clasificador cuando haya suficientes jugadores
    if not classifier.trained and len(player_dets) >= config.min_players_for_kmeans:
        logger.info(f"[TRAIN] Suficientes jugadores ({len(player_dets)}) para entrenar clasificador...")
        classifier.train_teams(frame, player_dets, config)
    
    if classifier.trained:
        logger.debug("[TRAIN] Clasificador ya entrenado")
    else:
        logger.debug(f"[TRAIN] En espera: {len(player_dets)}/{config.min_players_for_kmeans} jugadores")

    # Mapear posiciones de jugadores al tablero (sin IDs persistentes)
    players_frame: List[Tuple[int, np.ndarray, int]] = []
    if len(player_dets) > 0:
        logger.debug("[MAP] Mapeando posiciones de jugadores al tablero...")
        feet = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        for i, foot in enumerate(feet):
            dummy_id = i + 1  # Solo ID temporal para frame actual
            team_id = classifier.predict_team(frame, player_dets.xyxy[i])
            mapped = mapper.transform(foot)
            players_frame.append((dummy_id, mapped, team_id))
        
        # Estadísticas de equipos
        team_counts = {}
        for _, pos, team_id in players_frame:
            team_key = f"Team {team_id}" if team_id >= 0 else "Árbitro"
            team_counts[team_key] = team_counts.get(team_key, 0) + 1
        
        logger.info(f"[DETECT] {len(player_dets)} jugadores | Distribución: {team_counts}")

    # Procesar balón
    current_ball_pos: Optional[np.ndarray] = None
    if len(ball_dets) > 0:
        logger.debug("[BALL] Procesando detección de balón...")
        ball_center = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
        if field_validator.is_within_field(ball_center):
            current_ball_pos = mapper.transform(ball_center)
    
    # Actualizar rastreador de balón
    mapped_ball = ball_tracker.update(current_ball_pos)

    # Anotar frame de cámara
    logger.debug("[ANNOTATE] Anotando frame de cámara...")
    annotated = frame.copy()
    for i, bbox in enumerate(player_dets.xyxy):
        team_id = classifier.predict_team(frame, bbox)
        color = (128, 128, 128) if team_id < 0 else TacticalBoard.TEAM_COLORS[team_id % 2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        team_text = "ÁRBITRO" if team_id < 0 else f"EQUIPO {team_id}"
        text_size = cv2.getTextSize(team_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated, (x1, y1-30), (x1+text_size[0]+10, y1-5), color, -1)
        cv2.putText(annotated, team_text, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Dibujar balón
    if len(ball_dets) > 0:
        ball_center = ball_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
        bx, by = int(ball_center[0]), int(ball_center[1])
        cv2.circle(annotated, (bx, by), 8, (0, 165, 255), -1)
        cv2.circle(annotated, (bx, by), 9, (255, 255, 255), 2)

    # Renderizar tablero táctico
    logger.debug("[BOARD] Renderizando tablero táctico...")
    board_img = board_drawer.draw_state(players_frame, mapped_ball)

    logger.debug("[PROCESS] Frame procesado exitosamente")
    return annotated, board_img


# ===================== MAIN =====================

def main(cfg: Optional[Config] = None) -> None:
    """Pipeline principal de análisis."""
    cfg = cfg or Config()
    
    # Obtener input del usuario
    print("\n" + "="*70)
    print(" "*15 + "ANALIZADOR DE FUTSAL - CÁMARA LATERAL")
    print("="*70 + "\n")
    
    logger.info("["*10 + "STARTUP" + "]"*10)
    logger.info(f"Config: {cfg}")
    
    url = input("Ingresa URL de YouTube: ").strip()
    if not url:
        logger.error("URL vacía - abortando")
        return
    
    logger.info(f"URL ingresada: {url[:60]}...")

    start_time_str = input("Minuto de inicio (MM:SS, Enter para saltar): ").strip()
    logger.info(f"Tiempo de inicio: {start_time_str if start_time_str else 'Desde el principio'}")
    
    # Abrir stream
    logger.info("[STREAM] Abriendo stream de YouTube...")
    cap = open_youtube_stream(url, cfg)
    if cap is None or not cap.isOpened():
        logger.error("[ERROR] Falló al abrir stream")
        logger.error("  1. pip install --upgrade yt-dlp")
        logger.error("  2. Verifica que el video sea público")
        return
    
    logger.info("[OK] Stream abierto exitosamente")

    # Buscar minuto de inicio
    start_seconds = parse_start_time(start_time_str)
    if start_seconds > 0:
        logger.info(f"[SEEK] Buscando frame en {start_seconds}s...")
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info(f"[VIDEO] FPS: {fps}")

    # Leer primer frame
    logger.info("[FRAME] Leyendo primer frame del stream...")
    first_frame = read_first_frame(cap, cfg)
    if first_frame is None:
        logger.error("[ERROR] No se pudo leer el primer frame")
        cap.release()
        return
    
    frame_h, frame_w = first_frame.shape[:2]
    logger.info(f"[FRAME] Dimensiones del frame: {frame_w}x{frame_h}")

    # Calibración con rectángulo
    logger.info("[CALIBRATION] Abriendo interfaz de calibración...")
    calibrator = FieldCalibrator(first_frame.copy())
    field_rect = calibrator.calibrate()
    
    if len(field_rect) < 4:
        logger.error(f"[ERROR] Calibración falló: se necesitaban 4 puntos, se obtuvieron {len(field_rect)}")
        cap.release()
        return
    
    logger.info(f"[CALIBRATION] Calibración completada con {len(field_rect)} puntos")

    # Inicializar componentes
    logger.info("[INIT] Inicializando validador y mapeador de campo...")
    try:
        field_validator = FieldValidator(field_rect)
        logger.info("[OK] Validador de campo inicializado")
        
        mapper = SimpleFieldMapper(field_rect, cfg.board_width, cfg.board_height)
        logger.info("[OK] Mapeador de perspectiva inicializado")
    except Exception as e:
        logger.error(f"[ERROR] Error en setup: {e}")
        cap.release()
        return

    board_drawer = TacticalBoard(cfg.board_width, cfg.board_height)
    logger.info("Tablero táctico inicializado")
    
    classifier = TeamClassifier()
    logger.info("Clasificador de equipos inicializado")
    
    ball_tracker = BallTracker(max_distance=150)
    logger.info("Rastreador de balón inicializado")

    logger.info("Inicializando detector YOLO...")
    model = setup_detectors(cfg)
    if model is None:
        logger.error("FALLÓ: No se pudo cargar el modelo YOLO")
        cap.release()
        return
    
    logger.info("Detector YOLO cargado exitosamente")

    # Crear ventanas
    cv2.namedWindow("Vista Cámara", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tablero Táctico", cv2.WINDOW_NORMAL)

    frame_idx = 0
    t_start = time.time()
    logger.info("="*70)
    logger.info("ANÁLISIS EN TIEMPO REAL INICIADO - Presiona Q para detener")
    logger.info("(Sin tracking de IDs persistentes - detección en vivo)")
    logger.info("="*70 + "\n")

    try:
        # Bucle principal
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("[STREAM] Fin del stream alcanzado")
                break

            frame_idx += 1
            
            if frame_idx % 10 == 0:
                logger.info(f"[PROGRESS] Frame {frame_idx} - {time.time() - t_start:.1f}s transcurridos")

            # Procesar frame
            try:
                logger.debug(f"[FRAME {frame_idx}] Iniciando procesamiento...")
                annotated, board_img = process_frame(
                    frame, model, classifier, mapper,
                    field_validator, board_drawer, ball_tracker, cfg, fps
                )
                logger.debug(f"[FRAME {frame_idx}] Procesamiento completado")
            except Exception as e:
                logger.error(f"[ERROR] Frame {frame_idx}: {e}")
                continue

            # Overlay de contador
            elapsed = time.time() - t_start
            status_text = f"Frame {frame_idx} | {elapsed:.1f}s"
            cv2.putText(annotated, status_text,
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Mostrar
            cv2.imshow("Vista Cámara", annotated)
            cv2.imshow("Tablero Táctico", board_img)

            # Verificar salida
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("[USER] Análisis detenido por usuario")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("="*70)
        logger.info(f"ANÁLISIS COMPLETADO")
        logger.info(f"  - Frames procesados: {frame_idx}")
        logger.info(f"  - Tiempo total: {time.time() - t_start:.1f}s")
        logger.info(f"  - FPS promedio: {frame_idx / (time.time() - t_start):.1f}")
        logger.info("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario")
    except Exception as e:
        logger.exception(f"Error inesperado: {e}")
