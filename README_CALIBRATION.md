# Futsal Field Calibrator

Sistema de calibración de campo de futsal usando 6 puntos independientes con interfaz interactiva.

## Descripción

Este proyecto proporciona una herramienta simple e intuitiva para calibrar los límites del campo de futsal en videos de cámara lateral fija. Permite ajustar 6 puntos libres que forman un trapecio irregular, adaptándose a cualquier ángulo de cámara.

## Características

- ✅ **6 puntos independientes**: TL, CT, TR, BR, CB, BL (esquinas y centros)
- ✅ **Interfaz interactiva**: Arrastra puntos con el mouse
- ✅ **Canvas centrado**: Frame visualizado con padding para acceder fácilmente a todos los puntos
- ✅ **Visualización en tiempo real**: Vive el polígono mientras lo ajustas
- ✅ **Trapecio irregular**: Define el campo adaptándose a perspectivas complejas
- ✅ **Exportación de puntos**: Guarda los puntos calibrados en formato NumPy

## Requisitos

```bash
pip install opencv-python numpy
```

## Uso

### Opción 1: Archivo de video local
```bash
python calibracion_simple.py
# Ingresa: /ruta/a/tu/video.mp4
```

### Opción 2: URL de YouTube (requiere yt-dlp)
```bash
pip install yt-dlp
python calibracion_simple.py
# Ingresa: https://www.youtube.com/watch?v=VIDEO_ID
```

## Controles

| Control | Acción |
|---------|--------|
| **CLIC + ARRASTRAR** | Mover punto seleccionado |
| **ESPACIO** | Confirmar calibración |
| **R** | Resetear puntos a posición por defecto |
| **ESC** | Cancelar y salir |

## Puntos de Calibración

```
        1 (CT)
       /      \
      /        \
    0 (TL)    2 (TR)
    |          |
    |          |
    5 (BL)    3 (BR)
      \        /
       \      /
        4 (CB)
```

- **0 = TL (Top-Left)**: Esquina superior izquierda
- **1 = CT (Central Top)**: Centro de la línea superior
- **2 = TR (Top-Right)**: Esquina superior derecha
- **3 = BR (Bottom-Right)**: Esquina inferior derecha
- **4 = CB (Central Bottom)**: Centro de la línea inferior
- **5 = BL (Bottom-Left)**: Esquina inferior izquierda

## Salida

Después de completar la calibración:

1. **consola**: Coordenadas de los 6 puntos en formato readable
2. **`calibration_points.npy`**: Array NumPy con los puntos calibrados

```python
# Cargar puntos guardados
import numpy as np
points = np.load('calibration_points.npy')
# points.shape = (6, 2)
# points[0] = [x_TL, y_TL], etc.
```

## Ejemplo de flujo

```
1. Ejecuta: python calibracion_simple.py
2. Proporciona la fuente (video o URL)
3. Espera a que se cargue el primer frame
4. Ventana "Calibración de Campo" aparece
5. Arrastra los 6 puntos amarillos a los límites del campo
6. Presiona ESPACIO para confirmar
7. Los puntos se guardan en "calibration_points.npy"
```

## Nota Importante

Los 6 puntos deben:
- ✓ Estar posicionados en los límites reales del campo
- ✓ Encerrar completamente el área de juego
- ✓ Formar un polígono cerrado (los puntos se conectan automáticamente)

## Autor

**Isaac Boqué**  
📧 isboque19@gmail.com

## Licencia

MIT License
