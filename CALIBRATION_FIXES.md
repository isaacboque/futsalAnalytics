## Correcciones Realizadas al Calibrador de Campo

### Problemas Solucionados:

1. **Interfaz No Reactiva a Clicks**
   - El problema era que los eventos del mouse se perdían por el manejo incorrecto del estado
   - **Solución**: Cambié el estado del mouse de variables locales a atributos de clase (`self.dragging`, `self.active_point`)
   - Esto asegura que el callback del mouse tenga acceso correcto al estado mediante la closure

2. **Agregados 6 Puntos de Control**
   - Antes: 4 puntos (solo esquinas)
   - Ahora: 6 puntos (4 esquinas + 2 en la línea horizontal central)
   - **Esquinas**:
     - TL (Top-Left): arriba-izquierda
     - TR (Top-Right): arriba-derecha
     - BR (Bottom-Right): abajo-derecha
     - BL (Bottom-Left): abajo-izquierda
   - **Puntos Centrales**:
     - CL (Center-Left): centro-izquierda (en la línea Y central)
     - CR (Center-Right): centro-derecha (en la línea Y central)

### Cambios Técnicos:

#### FieldCalibrator
- ✅ Dibujo de 6 puntos en lugar de 4
- ✅ Línea horizontal central visible para referencia
- ✅ Métodos renombrados:
  - `get_corner_index()` → `get_point_index()` (ahora maneja 6 puntos)
  - `update_corner()` → `update_point()` (mejor lógica de actualización)
- ✅ Mejoras en el feedback visual (información en pantalla)
- ✅ Retorna 6 puntos en lugar de 4

#### FieldValidator
- ✅ Actualizado para aceptar 4-6 puntos
- ✅ Solo usa los primeros 4 puntos para geometría
- ✅ Los 2 puntos centrales se ignoran en la validación

#### SimpleFieldMapper
- ✅ Actualizado para aceptar 4-6 puntos
- ✅ Solo usa los primeros 4 puntos para perspectiva

#### Main Pipeline
- ✅ Cambió validación de "exactamente 4" a "al menos 4" puntos

### Ventajas de los 6 Puntos:

1. **Mejor Control Vertical**: Los puntos centrales permiten ajustar independientemente las alturas superior e inferior
2. **Mayor Precisión**: Especialmente útil si el campo no es perfectamente rectangular
3. **Intersección de Líneas**: Los puntos centrales marcan la intersección con la línea central del campo

### Cómo Usar:

1. Abre la interfaz de calibración
2. Verás 6 puntos amarillos: 4 en esquinas + 2 en la línea central
3. Haz clic y arrastra cada punto hacia los límites del campo
4. Los puntos ahora son **completamente reactivos**
5. Presiona ESPACIO para confirmar o ESC para cancelar
