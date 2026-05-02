"""Test de calibrador con 6 puntos completamente libres"""
import cv2
import numpy as np

# Copiar solo la clase FieldCalibrator actualizada
class FieldCalibrator:
    """
    Sistema de calibración por 6 puntos independientes para definir los límites del campo.
    Permite ajustar libremente cada punto para adaptarse al ángulo de cámara.
    """
    
    def __init__(self, frame: np.ndarray):
        """Inicializa el calibrador con 6 puntos independientes."""
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        
        # Inicializar 6 puntos como polígono: [TL, TR, BR, BL, CL, CR]
        # Puntos como array de coordenadas (x, y)
        margin_x = int(self.w * 0.1)
        margin_y = int(self.h * 0.15)
        cx = self.w // 2
        cy = self.h // 2
        
        self.points = np.array([
            [margin_x, margin_y],          # 0: TL (arriba-izquierda)
            [self.w - margin_x, margin_y], # 1: TR (arriba-derecha)
            [self.w - margin_x, self.h - margin_y], # 2: BR (abajo-derecha)
            [margin_x, self.h - margin_y], # 3: BL (abajo-izquierda)
            [margin_x, cy],                # 4: CL (centro-izquierda)
            [self.w - margin_x, cy],       # 5: CR (centro-derecha)
        ], dtype=np.float32)
        
        # Estados para el mouse
        self.dragging = False
        self.active_point = None
        self.WIN = "Calibración de Campo"
    
    def draw_frame(self) -> np.ndarray:
        """Dibuja el frame con el polígono y 6 puntos de control."""
        img = self.frame.copy()
        
        # Convertir puntos a enteros para dibujar
        pts = self.points.astype(np.int32)
        
        # Oscurecer área fuera del polígono
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), -1)
        cv2.fillPoly(overlay, [pts], (50, 100, 50))
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # Dibujar polígono con líneas
        for i in range(len(pts)):
            next_i = (i + 1) % len(pts)
            cv2.line(img, tuple(pts[i]), tuple(pts[next_i]), (0, 255, 0), 3)
        
        # Dibujar línea central (entre CL y CR)
        if len(pts) >= 6:
            cv2.line(img, tuple(pts[4]), tuple(pts[5]), (0, 200, 200), 2, cv2.LINE_AA)
        
        # Dibujar 6 puntos de control
        handle_size = 30
        thickness = 4
        labels = ["TL", "TR", "BR", "BL", "CL", "CR"]
        
        for idx, (pt, label) in enumerate(zip(pts, labels)):
            x, y = int(pt[0]), int(pt[1])
            
            # Color diferente para punto activo
            if self.active_point == idx and self.dragging:
                cv2.circle(img, (x, y), handle_size, (0, 165, 255), -1)  # Naranja
            else:
                cv2.circle(img, (x, y), handle_size, (0, 255, 255), -1)  # Amarillo
            
            cv2.circle(img, (x, y), handle_size, (255, 255, 255), thickness)  # Borde blanco
        
        # Información en pantalla
        info_text = "CLIC + ARRASTRAR puntos | ESPACIO: confirmar | R: resetear | ESC: cancelar"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return img
    
    def get_point_index(self, x: int, y: int):
        """Retorna el índice del punto más cercano si está dentro del threshold."""
        threshold = 40
        for idx, pt in enumerate(self.points):
            dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
            if dist < threshold:
                return idx
        return None
    
    def update_point(self, point_idx: int, x: int, y: int):
        """Actualiza la posición de un punto de control de forma completamente libre."""
        # Validar límites del frame con pequeño margen
        x = max(5, min(x, self.w - 5))
        y = max(5, min(y, self.h - 5))
        
        self.points[point_idx] = [x, y]
    
    def calibrate(self):
        """Ejecutar la interfaz de calibración."""
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
            """Maneja eventos del ratón de forma robusta."""
            if event == cv2.EVENT_LBUTTONDOWN:
                point_idx = self.get_point_index(x, y)
                if point_idx is not None:
                    self.dragging = True
                    self.active_point = point_idx
                    label = labels_map.get(point_idx, "?")
                    print(f"  → PUNTO {label} ({point_idx}) ACTIVO - Arrastrando...")
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging and self.active_point is not None:
                    self.update_point(self.active_point, x, y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if self.dragging and self.active_point is not None:
                    label = labels_map.get(self.active_point, "?")
                    print(f"  ✓ Punto {label} ({self.active_point}) - LIBERADO")
                self.dragging = False
                self.active_point = None
        
        # Registrar el callback
        cv2.setMouseCallback(self.WIN, mouse_event)
        
        print("Ventana abierta. Mueve los 6 puntos amarillos.\n")
        
        while True:
            img = self.draw_frame()
            cv2.imshow(self.WIN, img)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord(' '):
                print("✓ Calibración confirmada\n")
                break
            elif key == ord('r') or key == ord('R'):
                # Resetear a posición por defecto
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
                print("✓ Puntos reseteados\n")
            elif key == 27:  # ESC
                print("✗ Calibración cancelada\n")
                break
        
        cv2.destroyAllWindows()
        
        # Retornar los 6 puntos calibrados
        return self.points.copy()


# Test
print("=" * 70)
print("TEST DE CALIBRADOR - PUNTOS COMPLETAMENTE LIBRES")
print("=" * 70)

frame = np.zeros((600, 1000, 3), dtype=np.uint8)
cv2.rectangle(frame, (50, 50), (950, 550), (50, 50, 100), 2)
cv2.putText(frame, "Intenta mover los puntos libremente", (250, 300), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

calibrator = FieldCalibrator(frame)
points = calibrator.calibrate()

print("\nPuntos calibrados:")
labels = ["TL", "TR", "BR", "BL", "CL", "CR"]
for i, (label, point) in enumerate(zip(labels, points)):
    print(f"  {i+1}. {label}: ({int(point[0])}, {int(point[1])})")

print("\n✓ Test completado")
