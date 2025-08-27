import os
from tensorflow import keras

# Ruta del .h5 original
MODEL_DIR = "Model"
h5_path = os.path.join(MODEL_DIR, "Model/modelo_multitarea_final.h5")

# Ruta de salida .keras
keras_path = os.path.join(MODEL_DIR, "Model/modelo_multitarea_final.keras")

# Cargar el modelo antiguo
model = keras.models.load_model(h5_path, compile=False)

# Guardar en formato .keras
model.save(keras_path, save_format="keras")

print(f"✅ Modelo convertido y guardado en: {keras_path}")
