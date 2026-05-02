"""Test de integración del calibrador con 6 puntos libres"""
import cv2
import numpy as np

# Copiar clases necesarias del analyzer para la prueba
class FieldValidator:
    """Valida si los jugadores están dentro del campo."""
    
    def __init__(self, field_rect: np.ndarray):
        if len(field_rect) < 4:
            raise ValueError("Se necesitan al menos 4 puntos")
        
        corner_points = field_rect[:4]
        self.field_polygon = cv2.convexHull(corner_points.astype(np.float32))
        self.x_min = float(np.min(corner_points[:, 0]))
        self.x_max = float(np.max(corner_points[:, 0]))
        self.y_min = float(np.min(corner_points[:, 1]))
        self.y_max = float(np.max(corner_points[:, 1]))
        
        print(f"Validador inicializado con polígono de {len(field_rect)} puntos")
        print(f"  Límites: X[{self.x_min:.0f}, {self.x_max:.0f}], Y[{self.y_min:.0f}, {self.y_max:.0f}]")
    
    def is_within_field(self, point: np.ndarray) -> bool:
        if point[0] < self.x_min or point[0] > self.x_max or \
           point[1] < self.y_min or point[1] > self.y_max:
            return False
        result = cv2.pointPolygonTest(self.field_polygon, point[:2], False)
        return result >= 0


class SimpleFieldMapper:
    """Mapea coordenadas de cámara a tablero."""
    
    def __init__(self, field_rect: np.ndarray, board_width: int, board_height: int):
        src_pts = field_rect[:4].astype(np.float32)
        dst_pts = np.array([
            [0, 0],
            [board_width, 0],
            [board_width, board_height],
            [0, board_height]
        ], dtype=np.float32)
        
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.board_width = board_width
        self.board_height = board_height
        
        print(f"Mapeador inicializado: frame → tablero ({board_width}x{board_height})")
    
    def transform(self, pt: np.ndarray) -> np.ndarray:
        if pt[0] < 0 or pt[1] < 0:
            return pt
        pt_2d = pt.reshape(1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pt_2d, self.M)[0][0]
        return transformed


class FieldCalibrator:
    """Sistema de calibración por 6 puntos independientes."""
    
    def __init__(self, frame: np.ndarray):
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        
        margin_x = int(self.w * 0.1)
        margin_y = int(self.h * 0.15)
        cx = self.w // 2
        cy = self.h // 2
        
        self.points = np.array([
            [margin_x, margin_y],
            [self.w - margin_x, margin_y],
            [self.w - margin_x, self.h - margin_y],
            [margin_x, self.h - margin_y],
            [margin_x, cy],
            [self.w - margin_x, cy],
        ], dtype=np.float32)
        
        self.dragging = False
        self.active_point = None
        self.WIN = "Calibración de Campo"
    
    def draw_frame(self) -> np.ndarray:
        img = self.frame.copy()
        pts = self.points.astype(np.int32)
        
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), -1)
        cv2.fillPoly(overlay, [pts], (50, 100, 50))
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        for i in range(len(pts)):
            next_i = (i + 1) % len(pts)
            cv2.line(img, tuple(pts[i]), tuple(pts[next_i]), (0, 255, 0), 3)
        
        if len(pts) >= 6:
            cv2.line(img, tuple(pts[4]), tuple(pts[5]), (0, 200, 200), 2, cv2.LINE_AA)
        
        handle_size = 30
        thickness = 4
        labels = ["TL", "TR", "BR", "BL", "CL", "CR"]
        
        for idx, (pt, label) in enumerate(zip(pts, labels)):
            x, y = int(pt[0]), int(pt[1])
            
            if self.active_point == idx and self.dragging:
                cv2.circle(img, (x, y), handle_size, (0, 165, 255), -1)
            else:
                cv2.circle(img, (x, y), handle_size, (0, 255, 255), -1)
            
            cv2.circle(img, (x, y), handle_size, (255, 255, 255), thickness)
        
        info_text = "CLIC + ARRASTRAR puntos | ESPACIO: confirmar | R: resetear | ESC: cancelar"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return img
    
    def get_point_index(self, x: int, y: int):
        threshold = 40
        for idx, pt in enumerate(self.points):
            dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
            if dist < threshold:
                return idx
        return None
    
    def update_point(self, point_idx: int, x: int, y: int):
        x = max(5, min(x, self.w - 5))
        y = max(5, min(y, self.h - 5))
        self.points[point_idx] = [x, y]
    
    def calibrate(self) -> np.ndarray:
        print("\n" + "="*70)
        print("CALIBRACIÓN DE CAMPO - FUTSAL")
        print("="*70)
        print("\nInstrucciones:")
        print("  • 6 puntos de control: 4 esquinas + 2 en línea central")
        print("  • Arrastra cada punto libremente para adaptarse al ángulo de cámara")
        print("\nControles:")
        print("  - CLIC + ARRASTRAR en puntos para ajustar")
        print("  - ESPACIO para confirmar")
        print("  - R para resetear")
        print("  - ESC para cancelar")
        print("="*70 + "\n")
        
        cv2.namedWindow(self.WIN)
        
        labels_map = {0: "TL", 1: "TR", 2: "BR", 3: "BL", 4: "CL", 5: "CR"}
        
        def mouse_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point_idx = self.get_point_index(x, y)
                if point_idx is not None:
                    self.dragging = True
                    self.active_point = point_idx
                    label = labels_map.get(point_idx, "?")
                    print(f"  → PUNTO {label} ({point_idx}) activado")
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging and self.active_point is not None:
                    self.update_point(self.active_point, x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if self.dragging and self.active_point is not None:
                    label = labels_map.get(self.active_point, "?")
                    print(f"  ✓ Punto {label} liberado")
                self.dragging = False
                self.active_point = None
        
        cv2.setMouseCallback(self.WIN, mouse_event)
        print("Ventana abierta. Ajusta los 6 puntos.\n")
        
        while True:
            img = self.draw_frame()
            cv2.imshow(self.WIN, img)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord(' '):
                print("✓ Calibración confirmada\n")
                break
            elif key == ord('r') or key == ord('R'):
                margin_x = int(self.w * 0.1)
                margin_y = int(self.h * 0.15)
                cx = self.w // 2
                cy = self.h // 2
                
                self.points = np.array([
                    [margin_x, margin_y],
                    [self.w - margin_x, margin_y],
                    [self.w - margin_x, self.h - margin_y],
                    [margin_x, self.h - margin_y],
                    [margin_x, cy],
                    [self.w - margin_x, cy],
                ], dtype=np.float32)
                print("✓ Reseteado\n")
            elif key == 27:
                print("✗ Cancelado\n")
                break
        
        cv2.destroyAllWindows()
        return self.points.copy()


# TEST DE INTEGRACIÓN
print("=" * 70)
print("TEST DE INTEGRACIÓN - CALIBRADOR + VALIDATOR + MAPPER")
print("=" * 70 + "\n")

# Crear frame de prueba
frame = np.zeros((600, 1000, 3), dtype=np.uint8)
for y in range(0, 600, 50):
    cv2.line(frame, (0, y), (1000, y), (30, 30, 30), 1)
for x in range(0, 1000, 50):
    cv2.line(frame, (x, 0), (x, 600), (30, 30, 30), 1)

cv2.putText(frame, "Calibra el campo - Intenta hacer puntos irregulares", (200, 300), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

# Calibración
print("1. Ejecutando calibrador...")
calibrator = FieldCalibrator(frame)
field_rect = calibrator.calibrate()

print(f"\n2. Puntos obtenidos: {len(field_rect)} puntos")
for i, pt in enumerate(field_rect):
    print(f"   Punto {i}: ({int(pt[0])}, {int(pt[1])})")

# Validador
print("\n3. Inicializando validador...")
try:
    validator = FieldValidator(field_rect)
    print("   ✓ Validador OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Mapeador
print("\n4. Inicializando mapeador...")
try:
    mapper = SimpleFieldMapper(field_rect, 1000, 500)
    print("   ✓ Mapeador OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Pruebas
print("\n5. Testando validación y mapeo...")

# Probar puntos dentro y fuera
test_points = [
    (field_rect[0] + [10, 10], "Cerca de TL"),
    (field_rect[2] - [10, 10], "Cerca de BR"),
    (np.array([5, 5], dtype=np.float32), "Esquina del frame (fuera)"),
    (np.array([frame.shape[1]-5, frame.shape[0]-5], dtype=np.float32), "Bottom-right (fuera)"),
]

for point, label in test_points:
    inside = validator.is_within_field(point)
    mapped = mapper.transform(point)
    print(f"   {label}")
    print(f"     ¿Dentro?: {inside}")
    print(f"     Mapeado a tablero: ({int(mapped[0])}, {int(mapped[1])})")

print("\n" + "="*70)
print("✓ Test de integración completado exitosamente")
print("="*70)
