"""
Field Calibration System for Futsal
Sistema de calibración de campo para futsal usando 6 puntos libres.
"""

import cv2
import numpy as np
import logging
from typing import Optional

# ===================== LOGGING SETUP =====================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ===================== FIELD CALIBRATOR =====================

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
        
        # Dibujar línea central (entre CT y CB)
        if len(pts) >= 6:
            cv2.line(canvas, tuple(pts[1]), tuple(pts[4]), (0, 200, 200), 2, cv2.LINE_AA)
        
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
        print("  • El polígono define completamente el área de juego")
        print("\nCONTROLES:")
        print("  - CLIC + ARRASTRAR en puntos para ajustar")
        print("  - ESPACIO para confirmar")
        print("  - R para resetear")
        print("  - ESC para cancelar")
        print("="*70 + "\n")
        
        cv2.namedWindow(self.WIN)
        
        labels_map = {0: "TL", 1: "CT", 2: "TR", 3: "BR", 4: "CB", 5: "BL"}
        
        def mouse_event(event, x, y, flags, param):
            """Maneja eventos del ratón."""
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
        
        # Registrar el callback
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
        
        # Retornar los 6 puntos calibrados
        return self.points.copy()


# ===================== MAIN =====================

def main():
    """Ejecutar calibración del campo."""
    import sys
    
    print("\n" + "="*70)
    print(" "*20 + "FUTSAL FIELD CALIBRATOR")
    print("="*70 + "\n")
    
    # Obtener entrada del usuario
    source = input("Ingresa ruta de video o URL de YouTube: ").strip()
    if not source:
        logger.error("Entrada vacía - abortando")
        return
    
    logger.info(f"Fuente: {source}")
    
    # Abrir video/stream
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("No se pudo abrir el video/stream")
        return
    
    logger.info("Video abierto exitosamente")
    
    # Leer primer frame
    ret, frame = cap.read()
    if not ret:
        logger.error("No se pudo leer el primer frame")
        cap.release()
        return
    
    logger.info(f"Dimensiones del frame: {frame.shape[1]}x{frame.shape[0]}")
    
    # Calibración
    calibrator = FieldCalibrator(frame)
    calibrated_points = calibrator.calibrate()
    
    # Mostrar resultado
    print("\n" + "="*70)
    print("PUNTOS CALIBRADOS:")
    print("="*70)
    labels = ["TL", "CT", "TR", "BR", "CB", "BL"]
    for i, (pt, label) in enumerate(zip(calibrated_points, labels)):
        print(f"  {i} ({label:2s}): x={pt[0]:7.2f}, y={pt[1]:7.2f}")
    
    print("\n" + "="*70)
    print("Resultado en formato NumPy array:")
    print("="*70)
    print(f"\n{np.array_repr(calibrated_points)}")
    
    # Guardar puntos
    output_file = "calibration_points.npy"
    np.save(output_file, calibrated_points)
    print(f"\n[OK] Puntos guardados en: {output_file}")
    
    cap.release()
    print("\n[DONE] Calibración completada.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CANCEL] Cancelado por usuario")
    except Exception as e:
        logger.exception(f"Error: {e}")
