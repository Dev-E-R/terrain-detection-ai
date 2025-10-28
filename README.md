# 🤖 Sistema de Detección de Terreno con IA

Sistema de clasificación de terrenos mediante Deep Learning, diseñado como prueba de concepto para integración futura en robótica autónoma. Permite identificar diferentes tipos de superficies (rocosas, arenosas, con pasto, etc.) para navegación inteligente.

## 📋 Descripción del Proyecto

Este proyecto implementa una Red Neuronal Convolucional (CNN) utilizando TensorFlow/Keras para clasificar automáticamente tipos de terreno a partir de imágenes. La solución está pensada como componente de visión artificial para robots que necesitan identificar y adaptarse a diferentes superficies durante la navegación.

### Casos de Uso
- 🚗 Robots de navegación autónoma
- 🌍 Sistemas de exploración terrestre
- 🏗️ Vehículos de construcción inteligentes
- 🔍 Análisis automático de terreno en drones

## 🎯 Características Principales

- **Clasificación Multi-clase**: Detecta 4 tipos de terreno diferentes
- **Preprocesamiento Robusto**: Limpieza automática de imágenes corruptas
- **Arquitectura CNN Optimizada**: Modelo eficiente con data augmentation
- **Visualización Completa**: Análisis de resultados con matriz de confusión
- **Listo para Producción**: Guardado del modelo entrenado para despliegue

## 🏗️ Arquitectura del Modelo

### Red Neuronal Convolucional
```
Input (150x150x3)
    ↓
Conv2D(32) + MaxPooling
    ↓
Conv2D(64) + MaxPooling
    ↓
Conv2D(128) + MaxPooling
    ↓
Flatten + Dense(512)
    ↓
Output Dense(4) - Softmax
```

### Características Técnicas
- **Tamaño de entrada**: 150x150 píxeles RGB
- **Optimizador**: Adam
- **Función de pérdida**: Categorical Crossentropy
- **Data Augmentation**: Rotación, zoom, flip horizontal

## 📊 Dataset

**Fuente**: [Terrain Dataset en Kaggle](https://www.kaggle.com/datasets/ai21ds06anilriswal/terrain-dataset)

### Clases de Terreno
1. **Grassy Terrain** - Terreno con pasto
2. **Marshy Terrain** - Terreno pantanoso
3. **Rocky Terrain** - Terreno rocoso
4. **Sandy Terrain** - Terreno arenoso

### División del Dataset
- **Entrenamiento**: 80%
- **Validación**: 20%
- **Tamaño aproximado**: 239 MB

## 🚀 Instalación y Uso

### Requisitos Previos
```bash
Python 3.7+
Google Colab (recomendado) o Jupyter Notebook
```

### Dependencias
```python
tensorflow>=2.12.0
kagglehub
numpy
matplotlib
scikit-learn
Pillow
```

### Pasos para Ejecutar

1. **Abrir en Google Colab**
   - Sube el archivo `.ipynb` a Google Colab
   - O abre directamente desde GitHub

2. **Ejecutar Celdas Secuencialmente**
   ```python
   # 1. Descargar dataset
   # 2. Cargar imágenes
   # 3. Limpiar datos corruptos
   # 4. Entrenar modelo
   # 5. Evaluar resultados
   ```

3. **Descargar Modelo Entrenado**
   - El modelo se guarda automáticamente como `terrain_model.h5`

## 📈 Pipeline de Procesamiento

### 1. Descarga de Datos
```python
import kagglehub
path = kagglehub.dataset_download("ai21ds06anilriswal/terrain-dataset")
```

### 2. Carga de Imágenes
- Lectura desde directorio estructurado por clases
- Redimensionamiento a 150x150 píxeles
- Normalización de píxeles (0-1)

### 3. Limpieza de Datos
- Detección de imágenes corruptas
- Eliminación automática de archivos problemáticos
- Validación de integridad de datos

### 4. Entrenamiento
- **Épocas**: 15
- **Batch Size**: 32
- **Callbacks**: Early stopping si no mejora
- **Validación**: 20% del dataset

### 5. Evaluación
- Matriz de confusión
- Métricas por clase (precision, recall, f1-score)
- Visualización de predicciones

## 📊 Resultados Esperados

El modelo típicamente alcanza:
- **Accuracy en entrenamiento**: ~85-90%
- **Accuracy en validación**: ~80-85%
- **Tiempo de entrenamiento**: 10-15 minutos (GPU)

## 🗂️ Estructura del Proyecto

```
terrain-detection/
│
├── Copia_de_Untitled1.ipynb    # Notebook principal
├── README.md                     # Este archivo
├── requirements.txt              # Dependencias
├── terrain_model.h5             # Modelo entrenado (generado)
│
└── dataset/                     # Dataset descargado (automático)
    ├── Grassy_Terrain/
    ├── Marshy_Terrain/
    ├── Rocky_Terrain/
    └── Sandy_Terrain/
```

## 🔧 Integración con Robótica

### Uso del Modelo en Robot

```python
# Cargar modelo entrenado
from tensorflow.keras.models import load_model
model = load_model('terrain_model.h5')

# Capturar imagen desde cámara del robot
image = capture_from_robot_camera()
image = preprocess_image(image)  # Redimensionar a 150x150

# Predecir tipo de terreno
prediction = model.predict(image)
terrain_type = class_names[np.argmax(prediction)]

# Ajustar comportamiento del robot
adjust_robot_navigation(terrain_type)
```

## 🎨 Visualizaciones Incluidas

- ✅ Distribución de clases
- ✅ Curvas de aprendizaje (accuracy/loss)
- ✅ Matriz de confusión
- ✅ Ejemplos de predicciones correctas e incorrectas

## 🚧 Mejoras Futuras

- [ ] Ampliar a más tipos de terreno
- [ ] Implementar detección de objetos (piedras, obstáculos)
- [ ] Integración con ROS (Robot Operating System)
- [ ] Optimización para dispositivos embebidos (TensorFlow Lite)
- [ ] Detección en tiempo real con video
- [ ] Estimación de rugosidad del terreno

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Notas Técnicas

### Problemas Comunes y Soluciones

**Imágenes corruptas en el dataset**
- El código incluye limpieza automática
- Se eliminan archivos que PIL no puede abrir

**Memoria insuficiente**
- Usar Google Colab con GPU
- Reducir batch_size si es necesario
- Disminuir resolución de imágenes

**Overfitting**
- Data augmentation ya implementado
- Ajustar dropout si es necesario
- Aumentar dataset con más imágenes

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.

## 👤 Autor

Proyecto desarrollado como prueba de concepto para sistemas de navegación robótica inteligente.

## 🙏 Agradecimientos

- Dataset proporcionado por [ai21ds06anilriswal en Kaggle](https://www.kaggle.com/datasets/ai21ds06anilriswal/terrain-dataset)
- TensorFlow/Keras por el framework de Deep Learning
- Comunidad de Google Colab

---

**⚠️ Nota**: Este es un proyecto de demostración. Para implementación en producción, se recomienda validación adicional y pruebas en condiciones reales.

## 📞 Contacto

Para preguntas o sugerencias sobre el proyecto, abre un Issue en el repositorio.
