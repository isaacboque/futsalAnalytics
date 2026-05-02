"""Test simple del calibrador sin dependencias pesadas"""
import cv2
import numpy as np

# Copiar solo la clase FieldCalibrator
class FieldCalibrator:
    """
    Sistema de calibración por rectángulo ajustable para definir los límites del campo.
    Permite dibujar y ajustar un rectángulo que representa el área del futsal.
    Con 6 puntos: 4 en esquinas + 2 en línea central horizontal.
    """
    
    def __init__(self, frame: np.ndarray):
        """Inicializa el calibrador."""
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        
        # Inicializar rectángulo por defecto (área central del frame)
        margin = max(100, int(min(self.w, self.h) * 0.1))
        self.x1 = margin
        self.y1 = margin
        self.x2 = self.w - margin
        self.y2 = self.h - margin
        
        # Estados para el mouse
        self.dragging = False
        self.active_point = None
        self.WIN = "Calibración de Campo"
    
    def draw_frame(self) -> np.ndarray:
        """Dibuja el frame con el rectángulo y 6 puntos de control."""
        img = self.frame.copy()
        
        # Oscurecer área fuera del rectángulo
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (self.x1, self.y1), (self.x2, self.y2), (50, 100, 50), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # Dibujar rectángulo principal con línea MUY gruesa y visible
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 8)
        # Dibujar contorno adicional para mejor visibilidad
        cv2.rectangle(img, (self.x1-2, self.y1-2), (self.x2+2, self.y2+2), (255, 255, 255), 2)
        
        # Dibujar línea horizontal central
        cy_center = (self.y1 + self.y2) // 2
        cv2.line(img, (self.x1, cy_center), (self.x2, cy_center), (0, 200, 200), 3)
        
        # Dibujar 6 puntos de control (4 esquinas + 2 en línea central)
        handle_size = 30
        thickness = 4
        
        # 4 esquinas
        corners = [
            (self.x1, self.y1, "TL"),  # arriba-izquierda
            (self.x2, self.y1, "TR"),  # arriba-derecha
            (self.x2, self.y2, "BR"),  # abajo-derecha
            (self.x1, self.y2, "BL"),  # abajo-izquierda
        ]
        
        # 2 puntos en línea central
        cx_center = (self.x1 + self.x2) // 2
        center_points = [
            (self.x1, cy_center, "CL"),  # centro-izquierda
            (self.x2, cy_center, "CR"),  # centro-derecha
        ]
        
        all_points = corners + center_points
        
        for x, y, label in all_points:
            # Color diferente para punto activo
            if self.active_point is not None and self.dragging:
                if (label == "TL" and self.active_point == 0) or \
                   (label == "TR" and self.active_point == 1) or \
                   (label == "BR" and self.active_point == 2) or \
                   (label == "BL" and self.active_point == 3) or \
                   (label == "CL" and self.active_point == 4) or \
                   (label == "CR" and self.active_point == 5):
                    cv2.circle(img, (x, y), handle_size, (0, 165, 255), -1)  # Naranja cuando se arrastra
                else:
                    cv2.circle(img, (x, y), handle_size, (0, 255, 255), -1)  # Amarillo normal
            else:
                cv2.circle(img, (x, y), handle_size, (0, 255, 255), -1)  # Amarillo
            
            cv2.circle(img, (x, y), handle_size, (255, 255, 255), thickness)  # Borde blanco
        
        # Información en pantalla
        info_text = "CLIC + ARRASTRAR puntos | ESPACIO: confirmar | R: resetear | ESC: cancelar"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return img
    
    def get_point_index(self, x: int, y: int):
        """Retorna el índice del punto más cercano si está dentro del threshold."""
        cy_center = (self.y1 + self.y2) // 2
        
        points = [
            (self.x1, self.y1, 0),  # TL
            (self.x2, self.y1, 1),  # TR
            (self.x2, self.y2, 2),  # BR
            (self.x1, self.y2, 3),  # BL
            (self.x1, cy_center, 4),  # CL
            (self.x2, cy_center, 5),  # CR
        ]
        
        threshold = 40
        for px, py, idx in points:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < threshold:
                return idx
        
        return None
    
    def update_point(self, point_idx: int, x: int, y: int):
        """Actualiza la posición de un punto de control de forma independiente."""
        min_size = 50
        
        if point_idx == 0:  # arriba-izquierda
            self.x1 = max(0, min(x, self.x2 - min_size))
            self.y1 = max(0, min(y, self.y2 - min_size))
        elif point_idx == 1:  # arriba-derecha
            self.x2 = max(self.x1 + min_size, min(x, self.w))
            self.y1 = max(0, min(y, self.y2 - min_size))
        elif point_idx == 2:  # abajo-derecha
            self.x2 = max(self.x1 + min_size, min(x, self.w))
            self.y2 = max(self.y1 + min_size, min(y, self.h))
        elif point_idx == 3:  # abajo-izquierda
            self.x1 = max(0, min(x, self.x2 - min_size))
            self.y2 = max(self.y1 + min_size, min(y, self.h))
        elif point_idx == 4:  # centro-izquierda (x1 fijo, y libre)
            # Ajusta el límite más cercano a donde se arrastra
            cy_center = (self.y1 + self.y2) // 2
            if y < cy_center:
                self.y1 = max(0, min(y, self.y2 - min_size))
            else:
                self.y2 = max(self.y1 + min_size, min(y, self.h))
        elif point_idx == 5:  # centro-derecha (x2 fijo, y libre)
            # Ajusta el límite más cercano a donde se arrastra
            cy_center = (self.y1 + self.y2) // 2
            if y < cy_center:
                self.y1 = max(0, min(y, self.y2 - min_size))
            else:
                self.y2 = max(self.y1 + min_size, min(y, self.h))
    
    def calibrate(self):
        """Ejecutar la interfaz de calibración."""
        print("\n" + "="*70)
        print("CALIBRACIÓN DE CAMPO - FUTSAL")
        print("="*70)
        print("\nInstrucciones:")
        print("  • 6 puntos de control: 4 esquinas + 2 en línea central")
        print("  • Arrastra cada punto hacia los límites del campo")
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
                margin = max(100, int(min(self.w, self.h) * 0.1))
                self.x1 = margin
                self.y1 = margin
                self.x2 = self.w - margin
                self.y2 = self.h - margin
                print("✓ Rectángulo reseteado\n")
            elif key == 27:  # ESC
                print("✗ Calibración cancelada\n")
                break
        
        cv2.destroyAllWindows()
        
        # Retornar 6 puntos
        cy_center = (self.y1 + self.y2) // 2
        return np.array([
            [self.x1, self.y1],      # TL
            [self.x2, self.y1],      # TR
            [self.x2, self.y2],      # BR
            [self.x1, self.y2],      # BL
            [self.x1, cy_center],    # CL
            [self.x2, cy_center],    # CR
        ], dtype=np.float32)


# Test
print("=" * 70)
print("TEST DE CALIBRADOR")
print("=" * 70)

frame = np.zeros((600, 1000, 3), dtype=np.uint8)
cv2.rectangle(frame, (50, 50), (950, 550), (50, 50, 100), 2)

calibrator = FieldCalibrator(frame)
points = calibrator.calibrate()

print("\nPuntos calibrados:")
labels = ["TL", "TR", "BR", "BL", "CL", "CR"]
for i, (label, point) in enumerate(zip(labels, points)):
    print(f"  {i+1}. {label}: ({int(point[0])}, {int(point[1])})")

print("\n✓ Test completado")
