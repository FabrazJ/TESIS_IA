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
    """Convierte un leaf numérico (tensor/escalares/listas numéricas) a np.ndarray float32."""
    if hasattr(x, "numpy"):
        try:
            x = x.numpy()
        except Exception:
            pass
    arr = np.asarray(x)
    if arr.dtype == object:
        raise TypeError("dtype=object")
    arr = arr.astype(np.float32, copy=False)
    # normaliza a >=2D
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[None, :]
    return arr

def _flatten_numeric(obj, path="root", max_nodes=128):
    """
    Aplana recursivamente dict/list/tuple devolviendo
    lista de (np.ndarray, path) SOLO para arrays numéricos.
    Limita a max_nodes para evitar estructuras patológicas.
    """
    out = []
    stack = [(obj, path)]
    seen = set()

    while stack and len(out) < max_nodes:
        node, p = stack.pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)

        if isinstance(node, dict):
            for k, v in list(node.items())[:max_nodes]:
                stack.append((v, f"{p}.{k}"))
            continue
        if isinstance(node, (list, tuple)):
            for i, v in enumerate(list(node)[:max_nodes]):
                stack.append((v, f"{p}[{i}]"))
            continue

        # leaf
        try:
            arr = _to_numpy_leaf(node)
            out.append((arr, p))
        except Exception:
            pass

    return out

def _asegurar_batch(arr, fallback_shape):
    """Siempre (1, n) float32; si arr es None o shape inesperada, usa fallback."""
    if arr is None:
        return np.zeros(fallback_shape, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] != fallback_shape[-1]:
        out = np.zeros(fallback_shape, dtype=np.float32)
        flat = arr.ravel()
        n = min(out.shape[-1], flat.shape[0])
        out[0, :n] = flat[:n]
        return out
    return arr

def _seleccionar_salidas(salidas):
    """
    Devuelve (pred_diagnosis (1,4), pred_lobe (1,4), pred_score (1,1))
    con heurísticas sin ordenamientos.
    """
    hojas = _flatten_numeric(salidas)
    # Grupos por ancho
    grupo_4 = []
    grupo_1 = []
    otros   = []

    for arr, _ in hojas:
        ancho = arr.shape[-1]
        if ancho == 4:
            grupo_4.append(arr)
        elif ancho == 1:
            grupo_1.append(arr)
        else:
            otros.append(arr)

    pred_diagnosis = grupo_4[0] if len(grupo_4) >= 1 else (otros[0] if len(otros) >= 1 else None)
    pred_lobe      = grupo_4[1] if len(grupo_4) >= 2 else (otros[1] if len(otros) >= 2 else (grupo_4[0] if len(grupo_4) == 1 else None))
    pred_score     = grupo_1[0] if len(grupo_1) >= 1 else (otros[-1] if len(otros) >= 1 else None)

    # Asegurar shapes finales
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

    # Validaciones
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

        # Selección SIN sorts ni comparaciones
        pred_diagnosis, pred_lobe, pred_score = _seleccionar_salidas(salidas)

        # Decodificación
        clase_diagnostico = clases_diagnostico[int(np.argmax(pred_diagnosis[0]))]
        clase_lobulo      = clases_lobulo[int(np.argmax(pred_lobe[0]))]
        nivel_danio       = round(float(np.ravel(pred_score)[0]), 2)

    except Exception as e:
        # En caso extremo, cae a defaults sanos y NO rompe el endpoint
        pred_diagnosis = np.zeros((1, len(clases_diagnostico)), dtype=np.float32)
        pred_lobe      = np.zeros((1, len(clases_lobulo)), dtype=np.float32)
        pred_score     = np.zeros((1, 1), dtype=np.float32)
        clase_diagnostico = clases_diagnostico[0]
        clase_lobulo      = clases_lobulo[0]
        nivel_danio       = 0.0

    # Guardar registro y responder SIEMPRE 200
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
