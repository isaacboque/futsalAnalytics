#!/usr/bin/env python
"""Prueba del analyzer con URL de YouTube"""
import subprocess
import sys

url = "https://www.youtube.com/live/gW3EX3QS64s"
start_time = ""  # Sin minuto de inicio

print("=" * 70)
print("PRUEBA DE FUTSAL ANALYZER")
print("=" * 70)
print(f"\nURL: {url}")
print(f"Minuto: {start_time or 'Inicio del video'}\n")

# Crear entrada para el programa
input_data = f"{url}\n{start_time}\n"

# Ejecutar el programa
print("Iniciando futsal_analyzer.py...")
print("-" * 70)

try:
    proc = subprocess.Popen(
        [sys.executable, "futsal_analyzer.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=r"c:\Users\isboq\OneDrive\Escriptori\IAFS"
    )
    
    # Enviar entrada
    stdout, stderr = proc.communicate(input=input_data, timeout=60)
    
    if stdout:
        print("STDOUT:")
        print(stdout)
    
    if stderr:
        print("\nSTDERR:")
        print(stderr)
    
    print("-" * 70)
    print(f"Exit code: {proc.returncode}")
    
except subprocess.TimeoutExpired:
    print("✗ Timeout: El programa tardó más de 60 segundos")
    proc.kill()
except Exception as e:
    print(f"✗ Error: {e}")
