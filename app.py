import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np, csv
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.layers import InputLayer  # para compatibilidad con .h5

# ================== Config ==================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "Model")
MODEL_NAME = "modelo_multitarea_final.h5"      # puede ser .h5 o .keras
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

FILE_ID = os.getenv("DRIVE_FILE_ID", "1i8P8mkABFERZ-hBgz1Scpx_MjNxbAAwQ")
CSV_PATH = os.path.join(BASE_DIR, "registros_pacientes.csv")

clases_diagnostico = ['No Alzheimer', 'Alzheimer leve', 'Alzheimer moderado', 'Alzheimer severo']
clases_lobulo = ['Frontal', 'Temporal', 'Parietal', 'Occipital']

# ================== Helpers ==================
def ensure_model():
    """Descarga el modelo desde Google Drive si no existe localmente."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    import gdown  # gdown debe estar en requirements.txt
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("⏬ Descargando modelo desde Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No se pudo descargar el modelo desde Google Drive.")

def cargar_modelo():
    """Carga el modelo de forma robusta, compatible con InputLayer."""
    try:
        return keras.models.load_model(MODEL_PATH, compile=False)
    except TypeError as e:
        if "InputLayer" in str(e):
            return keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects={"InputLayer": InputLayer}
            )
        raise e

def preparar_imagen(archivo):
    try:
        img = Image.open(archivo).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error al procesar la imagen: {e}")

def guardar_en_csv(data):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(CSV_PATH) == 0:
            writer.writerow(['Fecha','Nombre','Cédula','Edad','Sexo','Diagnóstico','Lóbulo afectado','Nivel de daño'])
        writer.writerow(data)

# ================== App ==================
app = Flask(__name__)

# Garantiza el modelo local y cárgalo
ensure_model()
model = cargar_modelo()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    archivo = request.files.get('file')
    if not archivo or archivo.filename == '':
        return jsonify({'error': 'No se encontró archivo'}), 400

    nombre = request.form.get('nombre')
    cedula = request.form.get('cedula')
    edad   = request.form.get('edad')
    sexo   = request.form.get('sexo')

    try:
        img_array = preparar_imagen(archivo)

        # Espera 3 salidas: diagnóstico, lóbulo y score
        salidas = model.predict(img_array)
        if not isinstance(salidas, (list, tuple)) or len(salidas) < 3:
            return jsonify({'error': 'El modelo no retornó las 3 salidas esperadas.'}), 500

        pred_diagnosis, pred_lobe, pred_score = salidas
        clase_diagnostico = clases_diagnostico[int(np.argmax(pred_diagnosis[0]))]
        clase_lobulo      = clases_lobulo[int(np.argmax(pred_lobe[0]))]
        nivel_danio       = round(float(np.ravel(pred_score)[0]), 2)

        guardar_en_csv([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            nombre, cedula, edad, sexo,
            clase_diagnostico, clase_lobulo, nivel_danio
        ])

        return jsonify({
            'diagnostico': clase_diagnostico,
            'lobulo_afectado': clase_lobulo,
            'nivel_danio': nivel_danio
        })

    except Exception as e:
        return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500

@app.route('/registros', methods=['GET'])
def obtener_registros():
    try:
        if not os.path.isfile(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
            return jsonify([])
        with open(CSV_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            registros = [{k: (v or '') for k, v in row.items()} for row in reader]
        return jsonify(registros)
    except Exception as e:
        return jsonify({'error': f'Error al leer los registros: {str(e)}'}), 500

@app.route('/actualizar-registro', methods=['POST'])
def actualizar_registro():
    data = request.get_json()
    if not data or 'Cédula' not in data:
        return jsonify({'error': 'Datos inválidos'}), 400

    if not os.path.isfile(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return jsonify({'error': 'No hay registros para actualizar'}), 404

    cedula_objetivo = data['Cédula']
    actualizado = False
    registros = []

    with open(CSV_PATH, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        campos = reader.fieldnames or ['Fecha','Nombre','Cédula','Edad','Sexo','Diagnóstico','Lóbulo afectado','Nivel de daño']
        for fila in reader:
            if fila.get('Cédula') == cedula_objetivo:
                fila.update({k: str(v) for k, v in data.items()})
                actualizado = True
            registros.append(fila)

    if actualizado:
        with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=campos)
            writer.writeheader()
            writer.writerows(registros)
        return jsonify({'mensaje': 'Registro actualizado'})
    else:
        return jsonify({'error': 'Cédula no encontrada'}), 404

@app.route('/eliminar-registro', methods=['POST'])
def eliminar_registro():
    data = request.get_json()
    if not data or 'cedula' not in data:
        return jsonify({'error': 'Datos inválidos'}), 400

    if not os.path.isfile(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return jsonify({'error': 'No hay registros para eliminar'}), 404

    cedula_objetivo = str(data['cedula'])
    registros, eliminado = [], False

    with open(CSV_PATH, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        campos = reader.fieldnames or ['Fecha','Nombre','Cédula','Edad','Sexo','Diagnóstico','Lóbulo afectado','Nivel de daño']
        for fila in reader:
            if str(fila.get('Cédula', '')) != cedula_objetivo:
                registros.append(fila)
            else:
                eliminado = True

    if eliminado:
        with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=campos)
            writer.writeheader()
            writer.writerows(registros)
        return jsonify({'mensaje': 'Registro eliminado'})
    else:
        return jsonify({'error': 'Cédula no encontrada'}), 404


if __name__ == '__main__':
    # Para local: Render ignora esto porque usa gunicorn
    import os as _os
    app.run(host="0.0.0.0", port=int(_os.getenv("PORT", 5000)), debug=True)
