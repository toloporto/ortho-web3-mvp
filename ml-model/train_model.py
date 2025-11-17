# ml-model/train_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os

# Cargar datos sintéticos (reemplazar con reales después)
def load_synthetic_data():
    # Por ahora usaremos datos aleatorios
    x_train = np.random.random((100, 224, 224, 3))
    y_train = np.random.randint(0, 3, (100,))
    return x_train, y_train

def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation='softmax')  # 3 clases: I, II, III
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Entrenar modelo básico
x_train, y_train = load_synthetic_data()
model = create_model()
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
model.save('ortho_model.h5')