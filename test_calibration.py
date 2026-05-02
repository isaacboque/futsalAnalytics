#!/usr/bin/env python3
"""Script de prueba para la calibración interactiva."""

import cv2
import numpy as np
from futsal_analyzer import FieldCalibrator

# Crear un frame de prueba
test_frame = np.ones((600, 800, 3), dtype=np.uint8) * 150

# Crear calibrador
calibrator = FieldCalibrator(test_frame)

# Ejecutar calibración
print("\nIniciando prueba de calibración...")
print("Intenta hacer clic en los puntos amarillos y arrastrarlos.")
print("Los puntos deberían ser REACTIVOS a tus clicks.\n")

field_rect = calibrator.calibrate()

print(f"\n✓ Calibración completada")
print(f"Puntos obtenidos: {len(field_rect)}")
print("Puntos (esquinas + centros):")
for i, pt in enumerate(field_rect):
    labels = ["TL", "TR", "BR", "BL", "CL", "CR"]
    label = labels[i] if i < len(labels) else f"P{i}"
    print(f"  {label}: {pt}")
