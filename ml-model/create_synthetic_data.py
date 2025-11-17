# ml-model/create_synthetic_data.py
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def create_synthetic_dental_images():
    """Crea imágenes sintéticas de dientes para pruebas"""
    classes = ['class_i', 'class_ii', 'class_iii']
    for class_name in classes:
        os.makedirs(f'data/{class_name}', exist_ok=True)
        for i in range(100):  # 100 imágenes por clase
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            # Simular patrones básicos de dientes
            cv2.rectangle(img, (50, 80), (170, 120), (255, 255, 255), -1)
            cv2.imwrite(f'data/{class_name}/image_{i}.png', img)

create_synthetic_dental_images()