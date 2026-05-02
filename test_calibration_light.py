#!/usr/bin/env python3
"""Script de prueba ligero para calibración sin dependencias pesadas."""

import cv2
import numpy as np
from typing import Optional

class FieldCalibrator:
    """Prueba de calibración con 6 puntos."""
    
    def __init__(self, frame: np.ndarray):
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        
        margin = max(100, int(min(self.w, self.h) * 0.1))
        self.x1 = margin
        self.y1 = margin
        self.x2 = self.w - margin
        self.y2 = self.h - margin
        
        self.dragging = False
        self.active_point = None
        self.WIN = "Calibración de Campo"
    
    def draw_frame(self) -> np.ndarray:
        img = self.frame.copy()
        
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (self.x1, self.y1), (self.x2, self.y2), (50, 100, 50), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 5)
        
        cy_center = (self.y1 + self.y2) // 2
        cv2.line(img, (self.x1, cy_center), (self.x2, cy_center), (0, 200, 200), 2)
        
        handle_size = 25
        thickness = 3
        
        corners = [
            (self.x1, self.y1, "TL"),
            (self.x2, self.y1, "TR"),
            (self.x2, self.y2, "BR"),
            (self.x1, self.y2, "BL"),
        ]
        
        center_points = [
            (self.x1, cy_center, "CL"),
            (self.x2, cy_center, "CR"),
        ]
        
        all_points = corners + center_points
        
        for x, y, label in all_points:
            cv2.circle(img, (x, y), handle_size, (0, 255, 255), -1)
            cv2.circle(img, (x, y), handle_size, (255, 255, 255), thickness)
        
        info_text = "CLIC + ARRASTRAR puntos | ESPACIO: confirmar | R: resetear | ESC: cancelar"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return img
    
    def get_point_index(self, x: int, y: int) -> Optional[int]:
        cy_center = (self.y1 + self.y2) // 2
        
        points = [
            (self.x1, self.y1, 0),
            (self.x2, self.y1, 1),
            (self.x2, self.y2, 2),
            (self.x1, self.y2, 3),
            (self.x1, cy_center, 4),
            (self.x2, cy_center, 5),
        ]
        
        threshold = 40
        for px, py, idx in points:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < threshold:
                return idx
        
        return None
    
    def update_point(self, point_idx: int, x: int, y: int):
        min_size = 50
        cy_center = (self.y1 + self.y2) // 2
        
        if point_idx == 0:
            self.x1 = max(0, min(x, self.x2 - min_size))
            self.y1 = max(0, min(y, self.y2 - min_size))
        elif point_idx == 1:
            self.x2 = max(self.x1 + min_size, min(x, self.w))
            self.y1 = max(0, min(y, self.y2 - min_size))
        elif point_idx == 2:
            self.x2 = max(self.x1 + min_size, min(x, self.w))
            self.y2 = max(self.y1 + min_size, min(y, self.h))
        elif point_idx == 3:
            self.x1 = max(0, min(x, self.x2 - min_size))
            self.y2 = max(self.y1 + min_size, min(y, self.h))
        elif point_idx == 4:
            if y < cy_center:
                self.y1 = max(0, min(y, self.y2 - min_size))
            else:
                self.y2 = max(self.y1 + min_size, min(y, self.h))
        elif point_idx == 5:
            if y < cy_center:
                self.y1 = max(0, min(y, self.y2 - min_size))
            else:
                self.y2 = max(self.y1 + min_size, min(y, self.h))
    
    def calibrate(self) -> np.ndarray:
        print("\n" + "="*70)
        print("PRUEBA: Calibrador con 6 Puntos - Interfaz Reactiva")
        print("="*70)
        print("\n✓ Haz clic y arrastra los 6 puntos amarillos")
        print("✓ 4 esquinas + 2 en línea central")
        print("\nControles: ESPACIO=confirmar | R=resetear | ESC=cancelar\n")
        
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        
        def mouse_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point_idx = self.get_point_index(x, y)
                if point_idx is not None:
                    self.dragging = True
                    self.active_point = point_idx
                    labels = ["TL", "TR", "BR", "BL", "CL", "CR"]
                    print(f"  → Punto {labels[point_idx]} ({point_idx}) activado")
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging and self.active_point is not None:
                    self.update_point(self.active_point, x, y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if self.dragging:
                    labels = ["TL", "TR", "BR", "BL", "CL", "CR"]
                    print(f"  ← Punto {labels[self.active_point]} liberado")
                self.dragging = False
                self.active_point = None
        
        cv2.setMouseCallback(self.WIN, mouse_event)
        
        while True:
            img = self.draw_frame()
            cv2.imshow(self.WIN, img)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord(' '):
                print("\n✓ Calibración confirmada")
                break
            elif key == ord('r') or key == ord('R'):
                margin = max(100, int(min(self.w, self.h) * 0.1))
                self.x1 = margin
                self.y1 = margin
                self.x2 = self.w - margin
                self.y2 = self.h - margin
                print("✓ Rectángulo reseteado")
            elif key == 27:
                print("\n✗ Calibración cancelada")
                break
        
        cv2.destroyAllWindows()
        
        cy_center = (self.y1 + self.y2) // 2
        return np.array([
            [self.x1, self.y1],
            [self.x2, self.y1],
            [self.x2, self.y2],
            [self.x1, self.y2],
            [self.x1, cy_center],
            [self.x2, cy_center],
        ], dtype=np.float32)


# Crear frame de prueba
test_frame = np.ones((600, 800, 3), dtype=np.uint8) * 100

# Crear calibrador
calibrator = FieldCalibrator(test_frame)
field_rect = calibrator.calibrate()

print(f"\n✓ Calibración completada")
print(f"Puntos obtenidos: {len(field_rect)}")
print("Coordenadas:")
labels = ["TL", "TR", "BR", "BL", "CL", "CR"]
for i, pt in enumerate(field_rect):
    print(f"  {labels[i]}: ({pt[0]:.0f}, {pt[1]:.0f})")
