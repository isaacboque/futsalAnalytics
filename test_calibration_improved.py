"""Test mejorado de la interfaz de calibración"""
import cv2
import numpy as np
import sys
sys.path.insert(0, './')

from futsal_analyzer import FieldCalibrator

# Crear un frame de prueba (imagen negra con algunas características)
frame = np.zeros((600, 1000, 3), dtype=np.uint8)

# Agregar algunos patrones para que se vea menos vacío
cv2.rectangle(frame, (50, 50), (950, 550), (50, 50, 100), 2)
cv2.circle(frame, (500, 300), 100, (100, 100, 150), 2)
cv2.putText(frame, "Test Frame - Arrastra los puntos", (300, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

print("=" * 70)
print("PRUEBA DE CALIBRACIÓN MEJORADA")
print("=" * 70)
print("\nFrame creado. Abriendo interfaz de calibración...")
print("Intenta arrastrar los 6 puntos (esquinas y centros).")
print("\nControles:")
print("  - ARRASTRA los puntos amarillos")
print("  - Habrán 4 esquinas + 2 puntos en el centro")
print("  - ESPACIO para confirmar")
print("  - ESC para cancelar")
print("  - R para resetear")
print("\n" + "=" * 70 + "\n")

# Crear calibrador y ejecutar
calibrator = FieldCalibrator(frame)
points = calibrator.calibrate()

print("\n" + "=" * 70)
print("RESULTADO:")
print("=" * 70)
print(f"\nPuntos calibrados ({len(points)} total):")
labels = ["TL (Arriba-Izq)", "TR (Arriba-Dcha)", "BR (Abajo-Dcha)", 
          "BL (Abajo-Izq)", "CL (Centro-Izq)", "CR (Centro-Dcha)"]
for i, (label, point) in enumerate(zip(labels, points)):
    print(f"  {i+1}. {label:20} -> ({int(point[0]):4}, {int(point[1]):4})")

print("\n✓ Test completado exitosamente")
