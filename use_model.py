"""
Script de ejemplo para usar el modelo de detección de terreno entrenado
Útil para integración en sistemas robóticos
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Clases de terreno
CLASS_NAMES = ['Grassy_Terrain', 'Marshy_Terrain', 'Rocky_Terrain', 'Sandy_Terrain']

class TerrainDetector:
    """
    Detector de tipo de terreno usando el modelo entrenado
    """
    
    def __init__(self, model_path='terrain_model.h5'):
        """
        Inicializa el detector cargando el modelo
        
        Args:
            model_path (str): Ruta al archivo del modelo entrenado
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        print(f"Cargando modelo desde: {model_path}")
        self.model = load_model(model_path)
        self.img_size = (150, 150)
        print("✅ Modelo cargado exitosamente")
    
    def preprocess_image(self, img_path):
        """
        Preprocesa una imagen para el modelo
        
        Args:
            img_path (str): Ruta a la imagen
            
        Returns:
            numpy.ndarray: Imagen preprocesada
        """
        # Cargar imagen
        img = image.load_img(img_path, target_size=self.img_size)
        
        # Convertir a array
        img_array = image.img_to_array(img)
        
        # Normalizar píxeles
        img_array = img_array / 255.0
        
        # Añadir dimensión de batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, img_path, return_probabilities=False):
        """
        Predice el tipo de terreno de una imagen
        
        Args:
            img_path (str): Ruta a la imagen
            return_probabilities (bool): Si True, retorna probabilidades de todas las clases
            
        Returns:
            str o dict: Tipo de terreno predicho o diccionario con probabilidades
        """
        # Preprocesar imagen
        img_array = self.preprocess_image(img_path)
        
        # Hacer predicción
        predictions = self.model.predict(img_array, verbose=0)
        
        # Obtener clase predicha
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        if return_probabilities:
            # Retornar todas las probabilidades
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(CLASS_NAMES, predictions[0])
                }
            }
        else:
            return predicted_class
    
    def predict_batch(self, img_paths):
        """
        Predice el tipo de terreno para múltiples imágenes
        
        Args:
            img_paths (list): Lista de rutas a imágenes
            
        Returns:
            list: Lista de tipos de terreno predichos
        """
        results = []
        for img_path in img_paths:
            try:
                result = self.predict(img_path, return_probabilities=True)
                results.append(result)
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                results.append(None)
        
        return results
    
    def get_navigation_recommendation(self, terrain_type):
        """
        Proporciona recomendaciones de navegación según el terreno
        
        Args:
            terrain_type (str): Tipo de terreno detectado
            
        Returns:
            dict: Recomendaciones de navegación
        """
        recommendations = {
            'Grassy_Terrain': {
                'speed': 'normal',
                'traction': 'good',
                'caution': 'low',
                'notes': 'Terreno estable, velocidad normal'
            },
            'Marshy_Terrain': {
                'speed': 'slow',
                'traction': 'poor',
                'caution': 'high',
                'notes': 'Terreno inestable, reducir velocidad y evitar si es posible'
            },
            'Rocky_Terrain': {
                'speed': 'slow',
                'traction': 'variable',
                'caution': 'medium',
                'notes': 'Terreno irregular, ajustar suspensión y reducir velocidad'
            },
            'Sandy_Terrain': {
                'speed': 'moderate',
                'traction': 'moderate',
                'caution': 'medium',
                'notes': 'Terreno suelto, mantener momento y evitar paradas bruscas'
            }
        }
        
        return recommendations.get(terrain_type, {
            'speed': 'unknown',
            'traction': 'unknown',
            'caution': 'high',
            'notes': 'Terreno desconocido, proceder con precaución'
        })


# ============================================
# EJEMPLO DE USO
# ============================================

def main():
    """
    Ejemplo de uso del detector de terreno
    """
    print("=" * 50)
    print("🤖 DETECTOR DE TERRENO - DEMO")
    print("=" * 50)
    
    # Inicializar detector
    detector = TerrainDetector('terrain_model.h5')
    
    # Ejemplo 1: Predicción simple
    print("\n--- Ejemplo 1: Predicción Simple ---")
    img_path = 'ejemplo_terreno.jpg'
    
    # Nota: Descomentar cuando tengas una imagen de ejemplo
    # terrain = detector.predict(img_path)
    # print(f"Terreno detectado: {terrain}")
    
    # Ejemplo 2: Predicción con probabilidades
    print("\n--- Ejemplo 2: Predicción Detallada ---")
    # result = detector.predict(img_path, return_probabilities=True)
    # print(f"Clase predicha: {result['predicted_class']}")
    # print(f"Confianza: {result['confidence']:.2%}")
    # print("\nProbabilidades por clase:")
    # for clase, prob in result['probabilities'].items():
    #     print(f"  {clase}: {prob:.2%}")
    
    # Ejemplo 3: Recomendación de navegación
    print("\n--- Ejemplo 3: Recomendaciones de Navegación ---")
    for terrain in CLASS_NAMES:
        rec = detector.get_navigation_recommendation(terrain)
        print(f"\n{terrain}:")
        print(f"  Velocidad: {rec['speed']}")
        print(f"  Tracción: {rec['traction']}")
        print(f"  Precaución: {rec['caution']}")
        print(f"  Notas: {rec['notes']}")
    
    print("\n" + "=" * 50)
    print("✅ Demo completada")
    print("=" * 50)


# ============================================
# INTEGRACIÓN CON ROBOT (PSEUDO-CÓDIGO)
# ============================================

def robot_integration_example():
    """
    Ejemplo de cómo integrar el detector en un robot
    """
    # Inicializar detector
    detector = TerrainDetector('terrain_model.h5')
    
    # Bucle principal del robot
    while True:
        # 1. Capturar imagen desde cámara del robot
        # image = robot.camera.capture()
        # image.save('temp_terrain.jpg')
        
        # 2. Detectar tipo de terreno
        # result = detector.predict('temp_terrain.jpg', return_probabilities=True)
        # terrain = result['predicted_class']
        # confidence = result['confidence']
        
        # 3. Obtener recomendaciones
        # recommendations = detector.get_navigation_recommendation(terrain)
        
        # 4. Ajustar comportamiento del robot
        # if confidence > 0.8:  # Alta confianza
        #     if recommendations['speed'] == 'slow':
        #         robot.set_max_speed(0.3)  # 30% velocidad máxima
        #     elif recommendations['speed'] == 'moderate':
        #         robot.set_max_speed(0.6)  # 60% velocidad máxima
        #     else:
        #         robot.set_max_speed(1.0)  # 100% velocidad máxima
        #     
        #     if recommendations['caution'] == 'high':
        #         robot.enable_obstacle_avoidance(sensitivity='high')
        
        # 5. Log de telemetría
        # robot.log(f"Terrain: {terrain}, Confidence: {confidence:.2%}")
        
        pass  # Placeholder


if __name__ == '__main__':
    main()
