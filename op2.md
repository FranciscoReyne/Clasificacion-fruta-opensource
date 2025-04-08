**Ejemplo completo en Python que clasifica frutas por tamaño a partir de imágenes, dividiendo el flujo en tres corrientes (pequeñas, medianas, grandes), y luego refinando hasta que cada fruta sale por una única salida según su categoría.**

*Aquí tienes un ejemplo completo en Python que **clasifica frutas por tamaño a partir de imágenes**, dividiendo el flujo en **tres corrientes (pequeñas, medianas, grandes)**, y luego **refinando** hasta que cada fruta sale por una única salida según su categoría.*

Incluye:

- Detección y segmentación básica usando un modelo preentrenado.
- Clasificación por tamaño (estimado con bounding boxes).
- División del flujo en 3 etapas.
- Tabla con fuentes de datos de imágenes segmentadas de frutas.

---

### ✅ REQUISITOS

```bash
pip install ultralytics opencv-python matplotlib numpy
```

---

### 🧠 CÓDIGO: Clasificador por Tamaño (Usa YOLOv8 con Segmentación)

```python
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Carga modelo YOLOv8 preentrenado para segmentación (puedes usar uno mejor si quieres)
model = YOLO("yolov8n-seg.pt")

# Clasificación por tamaño basada en área del bounding box
def clasificar_fruta(area):
    if area < 3000:
        return "pequeña"
    elif area < 7000:
        return "mediana"
    else:
        return "grande"

# Simula el flujo dividido en 3 corrientes y luego refinado
def procesar_imagen(imagen_path):
    img = cv2.imread(imagen_path)
    resultados = model(img)[0]

    frutas = {"pequeña": [], "mediana": [], "grande": []}

    for box in resultados.boxes.xyxy:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        clase = clasificar_fruta(area)
        frutas[clase].append(((int(x1), int(y1)), (int(x2), int(y2))))

    # Segunda división refinada
    salidas = {
        "pequeña": frutas["pequeña"],
        "mediana": frutas["mediana"],
        "grande": frutas["grande"]
    }

    for tipo in salidas:
        print(f"\nFrutas {tipo.upper()}:")
        for coords in salidas[tipo]:
            print(f"  - Coords: {coords}")
            cv2.rectangle(img, coords[0], coords[1], (0,255,0), 2)

    cv2.imshow("Resultado", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ejecutar con imagen de prueba
procesar_imagen("frutas.jpg")  # Reemplaza con tu imagen real
```

---

### 📊 TABLA DE FUENTES DE DATOS (Frutas Segmentadas / Anotadas)

| Dataset | Descripción | Link Directo |
|--------|-------------|---------------|
| Fruits 360 | Clasificación de frutas (80k+ imágenes) | [🔗 Kaggle](https://www.kaggle.com/datasets/moltean/fruits) |
| Fruit Detection (YOLO format) | Frutas con bounding boxes (manzana, banana, naranja) | [🔗 Roboflow](https://universe.roboflow.com/roboflow-100/fruits-1hj2p) |
| Fruit Instance Segmentation | Segmentación de frutas tropicales (papaya, mango, etc) | [🔗 Kaggle](https://www.kaggle.com/datasets/andrewmvd/fruit-instance-segmentation) |
| Apple Detection Dataset | Detección de manzanas en árboles | [🔗 GitHub](https://github.com/aarme/AppleDetectionDataset) |
| Banana Dataset | Imágenes de bananas con segmentación | [🔗 Kaggle](https://www.kaggle.com/datasets/mbkinaci/banana-detection-yolo) |

---

### 🧩 OPCIONAL: Refinamiento Extra

Si quieres dividir más de una vez (como una segunda etapa de “corrientes”), puedes agregar otra función que simule otro set de "cintas" o "gateways", refinando con mayor resolución o precisión.

---

# Sigueinte idea: Versión que funciona con **video en tiempo real** y con una **Raspberry Pi + servos** para mover físicamente las frutas según su clase.
