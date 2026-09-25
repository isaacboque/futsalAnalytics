import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube
from sklearn.cluster import KMeans

# ===================== CONFIGURACIÓN GLOBAL =====================

MODEL_NAME = "yolo11n.pt"
BOARD_WIDTH = 800
BOARD_HEIGHT = 400
MAX_DISPLAY_WIDTH = 1280

# Número de puntos a seleccionar para calibrar (mínimo 6 para tu pieza a pieza)
NUM_CALIB_POINTS = 6  # puedes subirlo a 8, 10, etc.

# Clases de YOLO (0 = persona, 32 = balón en COCO)
PLAYER_CLASS_ID = 0
BALL_CLASS_ID = 32

MIN_PLAYERS_FOR_TRAIN = 8
MIN_PLAYERS_FOR_KMEANS = 6

# ===================== CLASIFICADOR DE EQUIPOS =====================

class TeamClassifier:
    """Separa jugadores en dos equipos según el color medio de la camiseta usando K-Means."""

    def __init__(self, n_clusters: int = 2, n_init: int = 10):
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        self.trained = False

    @staticmethod
    def _safe_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2]

    def get_jersey_color(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Devuelve el color medio BGR de la parte superior de la bbox (camiseta)."""
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        jersey_h = int(height * 0.4)
        crop = self._safe_crop(frame, x1, y1, x2, y1 + jersey_h)

        if crop.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)

        return np.mean(crop, axis=(0, 1)).astype(np.float32)

    def train_teams(self, frame: np.ndarray, detections: sv.Detections) -> None:
        """Entrena K-Means a partir de varias detecciones de jugadores."""
        if len(detections) < MIN_PLAYERS_FOR_KMEANS:
            return

        colors = [self.get_jersey_color(frame, bbox) for bbox in detections.xyxy]
        colors = np.array(colors, dtype=np.float32)

        # Evitar problemas si todos los colores son casi iguales
        if np.allclose(colors.max(axis=0), colors.min(axis=0)):
            return

        self.kmeans.fit(colors)
        self.trained = True
        print(">>> Clasificador de equipos entrenado correctamente.")

    def predict_team(self, frame: np.ndarray, bbox: np.ndarray) -> int:
        """Predice el ID del equipo (0 o 1)."""
        if not self.trained:
            return 0
        color = self.get_jersey_color(frame, bbox).reshape(1, -1)
        return int(self.kmeans.predict(color)[0])

# ===================== TRANSFORMACIÓN DEL CAMPO =====================

class PiecewiseTransformer:
    """
    Transforma coordenadas de la vista de cámara a un tablero 2D
    usando dos homografías (mitad izquierda y mitad derecha).
    """

    def __init__(self, src_pts: np.ndarray):
        if len(src_pts) != 6:
            raise ValueError("Se esperaban exactamente 6 puntos: TL, TR, BR, BL, MT, MB.")

        tl, tr, br, bl, mt, mb = src_pts
        mid_x = BOARD_WIDTH // 2

        # Homografía mitad izquierda
        src_left = np.array([tl, mt, mb, bl], dtype=np.float32)
        dst_left = np.array(
            [
                [0, 0],
                [mid_x, 0],
                [mid_x, BOARD_HEIGHT],
                [0, BOARD_HEIGHT],
            ],
            dtype=np.float32,
        )
        self.m_left = cv2.getPerspectiveTransform(src_left, dst_left)

        # Homografía mitad derecha
        src_right = np.array([mt, tr, br, mb], dtype=np.float32)
        dst_right = np.array(
            [
                [mid_x, 0],
                [BOARD_WIDTH, 0],
                [BOARD_WIDTH, BOARD_HEIGHT],
                [mid_x, BOARD_HEIGHT],
            ],
            dtype=np.float32,
        )
        self.m_right = cv2.getPerspectiveTransform(src_right, dst_right)

        # Línea media en la imagen de vídeo
        self.video_midline_x = float((mt[0] + mb[0]) / 2.0)

    def transform(self, pt: np.ndarray) -> np.ndarray:
        """Transforma un punto (x, y) de la imagen al plano del tablero."""
        x, _ = pt
        p = np.array([[pt]], dtype=np.float32)
        mat = self.m_left if x < self.video_midline_x else self.m_right
        mapped = cv2.perspectiveTransform(p, mat)
        return mapped[0][0]

# ===================== DIBUJO DEL TABLERO =====================

class TacticalBoard:
    """Genera una vista 2D simplificada del campo con jugadores y balón."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        # Team 0 = azul, Team 1 = amarillo
        self.team_colors = [(255, 0, 0), (0, 255, 255)]

    def _draw_pitch(self, board: np.ndarray) -> None:
        """Dibuja líneas principales del campo."""
        cv2.rectangle(board, (0, 0), (self.width, self.height), (255, 255, 255), 3)
        cv2.line(
            board,
            (self.width // 2, 0),
            (self.width // 2, self.height),
            (255, 255, 255),
            2,
        )
        cv2.circle(
            board,
            (self.width // 2, self.height // 2),
            50,
            (255, 255, 255),
            2,
        )

    def draw_state(
        self,
        players_mapped: list[tuple[np.ndarray, int]],
        ball_mapped: np.ndarray | None,
    ) -> np.ndarray:
        """Devuelve una imagen del tablero con el estado actual."""
        board = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        board[:] = (34, 139, 34)  # césped
        self._draw_pitch(board)

        # Jugadores
        for pos, team_id in players_mapped:
            x, y = int(pos[0]), int(pos[1])
            if team_id < 0 or team_id >= len(self.team_colors):
                team_id = 0
            color = self.team_colors[team_id]
            cv2.circle(board, (x, y), 10, color, -1)
            cv2.circle(board, (x, y), 11, (255, 255, 255), 1)

        # Balón
        if ball_mapped is not None:
            bx, by = int(ball_mapped[0]), int(ball_mapped[1])
            cv2.circle(board, (bx, by), 6, (0, 165, 255), -1)

        return board

# ===================== SELECCIÓN DE PUNTOS =====================

def select_points(frame: np.ndarray, num_points: int = NUM_CALIB_POINTS) -> np.ndarray:
    """
    Permite seleccionar manualmente puntos sobre el frame mostrando un número
    en el orden de clic.
    """
    if num_points <= 0:
        raise ValueError("num_points debe ser > 0")

    h, w = frame.shape[:2]
    scale = MAX_DISPLAY_WIDTH / w if w > MAX_DISPLAY_WIDTH else 1.0
    display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    pts: list[list[int]] = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < num_points:
            pts.append([int(x / scale), int(y / scale)])
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                display_frame, str(len(pts)), (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow("Calibrar Puntos", display_frame)

    print(
        "\nCalibración del campo:"
        f"\n- Haz clic en {num_points} puntos en el siguiente orden:"
        "\n  1: TL, 2: TR, 3: BR, 4: BL, 5: MID-TOP, 6: MID-BOTTOM"
        "\n  (si usas más puntos, distribúyelos por líneas y zonas clave del campo)"
    )

    cv2.imshow("Calibrar Puntos", display_frame)
    cv2.setMouseCallback("Calibrar Puntos", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(pts, dtype=np.float32)

# ===================== FUNCIÓN PRINCIPAL =====================

def main():
    url = input("YouTube URL: ").strip()
    if not url:
        print("URL vacía. Saliendo.")
        return

    start_time = input("Start Time (MM:SS, opcional): ").strip()

    try:
        cap = cap_from_youtube(url, '720p')
    except Exception as e:
        print(f"Error al abrir el vídeo de YouTube: {e}")
        return

    # Convertir tiempo a milisegundos
    seconds = 0
    if start_time:
        parts = start_time.split(':')
        try:
            if len(parts) == 2:
                minutes = int(parts[0])
                secs = int(parts[1])
                seconds = minutes * 60 + secs
            elif len(parts) == 1:
                seconds = int(parts[0])
        except ValueError:
            print("Formato de tiempo no válido, empezando desde 0.")

    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print("No se pudo leer el primer frame del vídeo.")
        return

    source_points = select_points(first_frame.copy(), NUM_CALIB_POINTS)
    if len(source_points) != 6:
        print("Se necesitan exactamente 6 puntos para el PiecewiseTransformer actual.")
        return

    try:
        transformer = PiecewiseTransformer(source_points)
    except Exception as e:
        print(f"Error creando el transformador de perspectiva: {e}")
        return

    board_drawer = TacticalBoard(BOARD_WIDTH, BOARD_HEIGHT)
    classifier = TeamClassifier()

    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Error cargando el modelo YOLO ({MODEL_NAME}): {e}")
        return

    tracker = sv.ByteTrack(track_activation_threshold=0.25)

    cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2D Tactical Board", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Detección
        results = model.predict(
            frame,
            classes=[PLAYER_CLASS_ID, BALL_CLASS_ID],
            conf=0.3,
            verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        # Jugadores y balón
        player_dets = detections[detections.class_id == PLAYER_CLASS_ID]
        ball_dets = detections[detections.class_id == BALL_CLASS_ID]

        tracked_players = tracker.update_with_detections(player_dets)

        # Entrenar clasificador
        if not classifier.trained and len(tracked_players) >= MIN_PLAYERS_FOR_TRAIN:
            classifier.train_teams(frame, tracked_players)

        # Mapear y clasificar jugadores
        players_to_draw: list[tuple[np.ndarray, int]] = []
        if len(tracked_players) > 0:
            feet_coords = tracked_players.get_anchors_coordinates(
                sv.Position.BOTTOM_CENTER
            )
            for i, foot in enumerate(feet_coords):
                mapped_pos = transformer.transform(foot)
                team_id = classifier.predict_team(frame, tracked_players.xyxy[i])
                players_to_draw.append((mapped_pos, team_id))

        # Mapear balón
        mapped_ball = None
        if len(ball_dets) > 0:
            ball_centers = ball_dets.get_anchors_coordinates(sv.Position.CENTER)
            mapped_ball = transformer.transform(ball_centers[0])

        # Visualización
        annotated_frame = frame.copy()
        for i, bbox in enumerate(tracked_players.xyxy):
            t_id = classifier.predict_team(frame, bbox)
            color = board_drawer.team_colors[t_id if 0 <= t_id < 2 else 0]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        board_img = board_drawer.draw_state(players_to_draw, mapped_ball)

        cv2.imshow("Camera View", annotated_frame)
        cv2.imshow("2D Tactical Board", board_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
