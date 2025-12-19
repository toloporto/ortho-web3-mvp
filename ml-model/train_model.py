# ml-model/train_model.py  (versión mejorada)
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2, preprocess_input
from tensorflow.keras import layers, callbacks

# Reproducibilidad (semillas)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 3
EPOCHS = 10

def load_synthetic_data(num_samples=100):
    # Datos sintéticos para pruebas: valores en 0-255 simulando imágenes
    x = np.random.randint(0, 256, size=(num_samples, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
    y = np.random.randint(0, NUM_CLASSES, size=(num_samples,), dtype=np.int32)
    return x, y

def make_dataset(x, y, training=True, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=1024, seed=SEED)
    # Convertir uint8 -> float32 y aplicar preprocess_input de MobileNetV2
    def _preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)  # MobileNetV2 expects inputs en rango [-1,1]
        return image, label
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def create_model(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES, base_trainable=False):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = base_trainable

    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = preprocess_input(x)  # Si ya preprocesas en dataset, puedes quitar esta línea
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    x, y = load_synthetic_data(num_samples=200)
    # Separar train/val
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    train_ds = make_dataset(x_train, y_train, training=True)
    val_ds = make_dataset(x_val, y_val, training=False)

    model = create_model(base_trainable=False)

    # Callbacks
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "best_model"),
        save_best_only=True,
        monitor="val_loss",
        save_weights_only=False,  # guardar todo el modelo (SavedModel)
        verbose=1
    )
    earlystop_cb = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr]
    )

    # Guardar en formato SavedModel (recomendado)
    model.save("ortho_model_saved", save_format="tf")

    # Opcional: fine-tuning (descongelar algunas capas)
    # example: unfreeze last N layers and recompile with smaller LR
    # base_model = model.layers[2]  # depende de la estructura; aquí es ilustrativo
    # base_model.trainable = True
    # for layer in base_model.layers[:-30]:
    #     layer.trainable = False
    # model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[checkpoint_cb, earlystop_cb])

if __name__ == "__main__":
    main()