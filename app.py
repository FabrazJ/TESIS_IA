import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import csv
from datetime import datetime
from tensorflow import keras

# ================== Config ==================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "Model")
MODEL_NAME = "modelo_multitarea_final.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
CSV_PATH   = os.path.join(BASE_DIR, "registros_pacientes.csv")
FILE_ID    = os.getenv("DRIVE_FILE_ID", "1i8P8mkABFERZ-hBgz1Scpx_MjNxbAAwQ")

clases_diagnostico = ['No Alzheimer', 'Alzheimer leve', 'Alzheimer moderado', 'Alzheimer severo']
clases_lobulo      = ['Frontal', 'Temporal', 'Parietal', 'Occipital']

# ================== Helpers ==================
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo no encontrado en {MODEL_PATH}. "
            "Debes subirlo al repositorio o al contenedor."
        )

model = None

def get_model():
    global model
    if model is None:
        ensure_model()
        model = keras.models.load_model(MODEL_PATH, compile=False)
    return model

def preparar_imagen(archivo):
    img = Image.open(archivo).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def guardar_en_csv(data):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(CSV_PATH) == 0:
            writer.writerow(['Fecha','Nombre','Cédula','Edad','Sexo','Diagnóstico','Lóbulo afectado','Nivel de daño'])
        writer.writerow(data)

# ---------- utilidades robustas ----------
def _to_numpy_leaf(x):
    """Convierte un 'leaf' (tensor, lista simple numérica, escalar) a np.ndarray numérico."""
    # Tensores TF
    if hasattr(x, "numpy"):
        try:
            x = x.numpy()
        except Exception:
            pass
    arr = np.asarray(x)
    if arr.dtype == object:
        raise TypeError("dtype=object")
    return arr.astype(np.float32, copy=False)

def _flatten_numeric(obj, path="root"):
    """
    Aplana recursivamente cualquier estructura y devuelve
    lista de (np.ndarray, path_str) SOLO para arrays numéricos.
    """
    out = []
    # dict
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten_numeric(v, f"{path}.{k}"))
        return out
    # list/tuple
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(_flatten_numeric(v, f"{path}[{i}]"))
        return out
    # leaf
    try:
        arr = _to_numpy_leaf(obj)
    except Exception:
        return []
    # normaliza a al menos 2D (batch x features)
    if arr.ndim == 0:   # escalar
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1: # vector
        arr = arr[None, :]
    return [(arr, path)]

def _asegurar_batch(arr, fallback_shape):
    """Siempre (1, n) float32; si arr es None o shape inesperada, usa fallback."""
    if arr is None:
        return np.zeros(fallback_shape, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[None, :]
    # Ajuste de columnas si no coincide
    if arr.shape[-1] != fallback_shape[-1]:
        out = np.zeros(fallback_shape, dtype=np.float32)
        flat = arr.ravel()
        n = min(out.shape[-1], flat.shape[0])
        out[0, :n] = flat[:n]
        return out
    return arr

def _seleccionar_salidas(salidas):
    """
    A partir de cualquier estructura, determina:
      - pred_diagnosis: (1,4)
      - pred_lobe:      (1,4)
      - pred_score:     (1,1)
    usando heurísticas seguras si faltan formas exactas.
    """
    hojas = _flatten_numeric(salidas)  # [(arr, path), ...]
    if not hojas:
        return None, None, None

    # Candidatos por forma
    cand_4 = [(a, p) for (a, p) in hojas if a.shape[-1] == 4]
    cand_1 = [(a, p) for (a, p) in hojas if a.shape[-1] == 1]

    pred_diagnosis = None
    pred_lobe      = None
    pred_score     = None

    # Preferimos exactamente dos (1,4)
    if len(cand_4) >= 2:
        # Ordena por "confianza" heurística: mayor varianza en eje de clases
        cand_4.sort(key=lambda t: float(np.var(t[0])), reverse=True)
        pred_diagnosis = cand_4[0][0]
        pred_lobe      = cand_4[1][0]
    elif len(cand_4) == 1:
        pred_diagnosis = cand_4[0][0]

    # Para score preferimos (1,1)
    if len(cand_1) >= 1:
        pred_score = cand_1[0][0]

    # Si aún falta alguno, usa heurística por “ancho” (más clases)
    if pred_diagnosis is None or pred_lobe is None:
        # Ordena por ancho de la última dimensión (desc) y varianza
        resto = [(a, p) for (a, p) in hojas]
        resto.sort(key=lambda t: (t[0].shape[-1], float(np.var(t[0]))), reverse=True)
        # Toma los dos primeros como clasificaciones
        if pred_diagnosis is None and len(resto) >= 1:
            pred_diagnosis = resto[0][0]
        if pred_lobe is None and len(resto) >= 2:
            pred_lobe = resto[1][0]

    # Si falta score, intenta algún (1,1); si no, cualquier escalar/lo más chico
    if pred_score is None:
        # busca el más pequeño en ancho
        hojas_por_ancho = sorted(hojas, key=lambda t: t[0].shape[-1])
        pred_score = hojas_por_ancho[0][0]

    # Asegura shapes finales
    pred_diagnosis = _asegurar_batch(pred_diagnosis, (1, len(clases_diagnostico)))
    pred_lobe      = _asegurar_batch(pred_lobe,      (1, len(clases_lobulo)))
    pred_score     = _asegurar_batch(pred_score,     (1, 1))

    return pred_diagnosis, pred_lobe, pred_score

# ================== App ==================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# ================== Función Predict ==================
@app.route('/predict', methods=['POST'])
def predict():
    archivo = request.files.get('file')
    if not archivo or archivo.filename == '':
        return jsonify({'error': 'No se encontró archivo'}), 400

    nombre = request.form.get('nombre', '').strip()
    cedula = request.form.get('cedula', '').strip()
    edad   = request.form.get('edad', '').strip()
    sexo   = request.form.get('sexo', '').strip()

    # Validaciones básicas
    if not nombre or not cedula or not edad or not sexo:
        return jsonify({'error': 'Todos los campos son obligatorios'}), 400
    if not cedula.isdigit():
        return jsonify({'error': 'La cédula debe contener solo números'}), 400
    try:
        edad = int(edad)
        if edad < 0 or edad > 120:
            return jsonify({'error': 'Edad inválida'}), 400
    except Exception:
        return jsonify({'error': 'Edad inválida'}), 400

    # Procesar imagen
    try:
        img_array = preparar_imagen(archivo)
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 400

    # Predicción robusta
    try:
        model_obj = get_model()
        salidas = model_obj.predict(img_array)

        # Selección automática de salidas
        pred_diagnosis, pred_lobe, pred_score = _seleccionar_salidas(salidas)

        # Decodificación
        clase_diagnostico = clases_diagnostico[int(np.argmax(pred_diagnosis[0]))]
        clase_lobulo      = clases_lobulo[int(np.argmax(pred_lobe[0]))]
        nivel_danio       = round(float(np.ravel(pred_score)[0]), 2)

        # Guardar registro
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
        return jsonify({'error': f'Error en la predicción o modelo: {str(e)}'}), 500

# ================== Gestión de registros ==================
@app.route('/registros', methods=['GET'])
def obtener_registros():
    if not os.path.isfile(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
        return jsonify([])
    with open(CSV_PATH, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        registros = [{k: (v or '') for k, v in row.items()} for row in reader]
    return jsonify(registros)

@app.route('/actualizar-registro', methods=['POST'])
def actualizar_registro():
    data = request.get_json()
    if not data or 'Cédula' not in data:
        return jsonify({'error': 'Datos inválidos'}), 400
    if not os.path.isfile(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return jsonify({'error': 'No hay registros para actualizar'}), 404

    cedula_objetivo = str(data['Cédula'])
    registros, actualizado = [], False

    with open(CSV_PATH, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        campos = reader.fieldnames or ['Fecha','Nombre','Cédula','Edad','Sexo','Diagnóstico','Lóbulo afectado','Nivel de daño']
        for fila in reader:
            if str(fila.get('Cédula', '')) == cedula_objetivo:
                fila.update({k: str(v) for k, v in data.items()})
                actualizado = True
            registros.append(fila)

    if actualizado:
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=campos)
            writer.writeheader()
            writer.writerows(registros)
        return jsonify({'mensaje': 'Registro actualizado'})
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

    with open(CSV_PATH, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        campos = reader.fieldnames or ['Fecha','Nombre','Cédula','Edad','Sexo','Diagnóstico','Lóbulo afectado','Nivel de daño']
        for fila in reader:
            if str(fila.get('Cédula', '')) != cedula_objetivo:
                registros.append(fila)
            else:
                eliminado = True

    if eliminado:
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=campos)
            writer.writeheader()
            writer.writerows(registros)
        return jsonify({'mensaje': 'Registro eliminado'})
    return jsonify({'error': 'Cédula no encontrada'}), 404

# ================== Run App ==================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
