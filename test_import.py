#!/usr/bin/env python
"""Test de importación del analyzer"""
import sys
sys.path.insert(0, r'c:\Users\isboq\OneDrive\Escriptori\IAFS')

print("Intentando importar futsal_analyzer...")
try:
    from futsal_analyzer import FieldCalibrator, Config
    print("✓ Import FieldCalibrator OK")
    print("✓ Import Config OK")
    
    import numpy as np
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cal = FieldCalibrator(frame)
    print("✓ FieldCalibrator instantiation OK")
    
    print("\n✓ Módulo futsal_analyzer funciona correctamente")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
