"""
==========================================================================================
 PRÁCTICA 3 — DIVERSIDAD EN RECEPCIÓN (MRC, Maximal Ratio Combining)
==========================================================================================
 Archivo AUTOCONTENIDO: no importa nada de app.py ni de los otros archivos de práctica.
 El código repetido respecto a los demás archivos es a propósito, para poder explicar esta
 práctica leyendo un solo archivo.

 ORDEN DEL ARCHIVO (de arriba hacia abajo):
   1) LÓGICA DE LA PRÁCTICA  -> lo importante a explicar: un transmisor OFDM y un receptor
                                con n_rx antenas que combina con MRC (vs. 1 antena con ZF).
   2) SOPORTE                -> modulaciones QAM, parámetros LTE, canal, ruido, BER, imagen.
   3) MONTE CARLO + FLASK    -> generación de curvas y endpoints web (no es el foco).

 Idea de la diversidad en RX (Práctica 3): el MISMO símbolo OFDM llega por n_rx antenas, cada
 una con su propio canal Pedestrian A independiente y su propio ruido. Combinando las antenas
 con MRC (pondera cada una por conj(H_m) y la alinea en fase) sube la SNR efectiva y baja la
 BER. Con n_rx = 1 el receptor es el clásico Zero-Forcing (Y/H); con n_rx > 1 hay ganancia de
 diversidad:  X_est[k] = Σ_m conj(H_m[k])·Y_m[k] / Σ_m |H_m[k]|².
==========================================================================================
"""

import os
import io
import time
import base64
import math
from typing import Dict, Tuple

import numpy as np
from scipy.stats import t as t_student
from PIL import Image
from flask import Flask, render_template, request, jsonify


# =====================================================================================
# ===              1) LÓGICA DE LA PRÁCTICA — DIVERSIDAD EN RX (MRC)                 ===
# =====================================================================================
# Parte central de la práctica. El transmisor es OFDM normal; lo nuevo está en el receptor:
# combinar varias antenas con MRC. Las funciones OFDM (pilotos, IFFT/FFT + CP) son el soporte
# de transmisión que el receptor MRC necesita.

PASO_PILOTO = 12  # En OFDM se inserta 1 subportadora piloto cada 12 subportadoras


def indices_pilotos_datos(n_sc: int, paso: int = PASO_PILOTO) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve los índices de las subportadoras piloto y de datos (OFDM)."""
    indices = np.arange(n_sc)
    pilotos = indices[::paso]
    datos = np.setdiff1d(indices, pilotos, assume_unique=False)
    return pilotos, datos


def insertar_pilotos(simbolos_datos: np.ndarray, n_sc: int,
                     paso: int = PASO_PILOTO, valor_piloto: complex = 1 + 0j) -> np.ndarray:
    """Arma la rejilla OFDM: datos + pilotos intercalados cada `paso`."""
    pilotos, datos = indices_pilotos_datos(n_sc, paso)
    rejilla = np.zeros(n_sc, dtype=complex)
    rejilla[pilotos] = valor_piloto
    n = min(len(datos), len(simbolos_datos))
    rejilla[datos[:n]] = simbolos_datos[:n]
    return rejilla


def mapeo_sc_a_fft(rejilla_sc: np.ndarray, n_fft: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mapea las N_SC subportadoras a los N_FFT puntos en banda base (DC vacía, ±centrado)."""
    n_sc = len(rejilla_sc)
    mitad = n_sc // 2
    indices_pos = np.arange(1, mitad + 1)
    indices_neg = n_fft - np.arange(1, mitad + 1)[::-1]
    indices_fft = np.concatenate([indices_neg, indices_pos])
    vec = np.zeros(n_fft, dtype=complex)
    vec[indices_fft] = rejilla_sc
    return vec, indices_fft


def modulacion_ofdm(rejilla_sc: np.ndarray, n_fft: int, n_cp: int) -> Tuple[np.ndarray, np.ndarray]:
    """IFFT + prefijo cíclico: convierte la rejilla de subportadoras en señal de tiempo."""
    vec_freq, indices_fft = mapeo_sc_a_fft(rejilla_sc, n_fft)
    senal_t = np.fft.ifft(vec_freq) * n_fft / np.sqrt(len(rejilla_sc))
    cp = senal_t[-n_cp:] if n_cp > 0 else np.array([], dtype=complex)
    return np.concatenate([cp, senal_t]), indices_fft


def demodulacion_ofdm(simbolo_con_cp: np.ndarray, indices_fft: np.ndarray,
                       n_fft: int, n_cp: int, n_sc: int) -> np.ndarray:
    """Quita el CP, aplica la FFT y extrae las N_SC subportadoras activas."""
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)
    return espectro[indices_fft]


def cadena_tx_rx(bits_tx: np.ndarray, modulacion: str, params: Dict,
                  snr_db: float, rng: np.random.Generator,
                  capturar_constelaciones: bool = False, n_rx: int = 1) -> Dict:
    """
    NÚCLEO DE LA PRÁCTICA 3. Transmisor OFDM y receptor de n_rx antenas:

      - n_rx == 1 -> receptor clásico de 1 antena con ecualización Zero-Forcing (Y/H).
      - n_rx  > 1 -> el MISMO símbolo TX llega por n_rx canales independientes, cada uno con su
                     propio ruido, y se combinan con MRC (Maximal Ratio Combining):
                       X_est[k] = Σ_m conj(H_m[k])·Y_m[k] / Σ_m |H_m[k]|².
                     Al aumentar n_rx mejora la SNR efectiva y la BER baja (diversidad).

    Devuelve bits recibidos, BER y, opcionalmente, las constelaciones RX (con n_rx>1 también la
    de una sola antena "antes" de combinar, para comparar).
    """
    mod = MODULACIONES[modulacion]
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    n_datos = params["n_datos"]
    fs = params["fs"]

    # --- TX: bits -> símbolos QAM (con relleno a múltiplos exactos) ---
    bps = mod["bits"]
    falta = (-len(bits_tx)) % bps
    if falta:
        bits_tx_pad = np.concatenate([bits_tx, np.zeros(falta, dtype=np.uint8)])
    else:
        bits_tx_pad = bits_tx
    simbolos_qam = mod["mapear"](bits_tx_pad)

    falta_sim = (-len(simbolos_qam)) % n_datos
    if falta_sim:
        simbolos_qam = np.concatenate([simbolos_qam, np.zeros(falta_sim, dtype=complex)])
    n_ofdm = len(simbolos_qam) // n_datos

    constelacion_tx = simbolos_qam.copy() if capturar_constelaciones else None
    simbolos_rx_datos = []                            # Símbolos RX combinados (constelación)
    simbolos_rx_antes = []                            # Símbolos RX de 1 antena (antes de MRC)
    bits_rx_total = []
    _, indices_datos_sc = indices_pilotos_datos(n_sc)

    for i in range(n_ofdm):
        bloque_datos = simbolos_qam[i * n_datos:(i + 1) * n_datos]
        rejilla = insertar_pilotos(bloque_datos, n_sc)              # TX OFDM: datos + pilotos
        _, indices_fft = modulacion_ofdm(rejilla, n_fft, n_cp)      # IFFT + CP (solo para indices_fft)

        if n_rx <= 1:
            # --- Receptor de 1 antena: canal Pedestrian A + ecualización Zero-Forcing ---
            H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)  # Respuesta del canal H[k]
            senal_canal, _ = modulacion_ofdm(rejilla * H, n_fft, n_cp)   # Y[k]=H[k]·X[k], al tiempo
            senal_rx = agregar_ruido_awgn(senal_canal, snr_db, rng)      # AWGN
            rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)  # CP + FFT
            rejilla_eq = rejilla_rx / H                                  # Ecualización ZF
        else:
            # --- Diversidad en RX con MRC: n_rx antenas, cada una con canal y ruido propios ---
            num = np.zeros(n_sc, dtype=complex)         # Numerador MRC: Σ conj(H_m)·Y_m
            den = np.zeros(n_sc, dtype=float)           # Denominador MRC: Σ |H_m|²
            rejilla_eq_antes = None                     # Estimación con 1 sola antena (referencia "antes")
            for m in range(n_rx):
                H_m = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)  # Canal independiente
                senal_canal, _ = modulacion_ofdm(rejilla * H_m, n_fft, n_cp)   # Y_m[k]=H_m[k]·X[k]
                senal_rx = agregar_ruido_awgn(senal_canal, snr_db, rng)        # Ruido independiente
                rejilla_rx_m = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
                num += np.conj(H_m) * rejilla_rx_m       # Pondera y alinea en fase (combinación coherente)
                den += np.abs(H_m) ** 2                  # Suma de ganancias de canal
                if m == 0:
                    rejilla_eq_antes = rejilla_rx_m / H_m  # ZF de la 1ª antena (sin diversidad)
            rejilla_eq = num / (den + 1e-12)            # Estimación MRC de X[k]
            if capturar_constelaciones:
                simbolos_rx_antes.append(rejilla_eq_antes[indices_datos_sc])

        datos_rx = rejilla_eq[indices_datos_sc]                      # Posiciones de datos
        if capturar_constelaciones:
            simbolos_rx_datos.append(datos_rx)
        bits_rx_total.append(mod["demapear"](datos_rx))

    bits_rx = np.concatenate(bits_rx_total) if bits_rx_total else np.array([], dtype=np.uint8)
    bits_rx = bits_rx[:len(bits_tx_pad)]
    ber = calcular_ber(bits_tx_pad, bits_rx)

    salida = {
        "bits_rx": bits_rx[:len(bits_tx)],
        "ber": ber,
        "n_simbolos_ofdm": n_ofdm,
    }
    if capturar_constelaciones:
        salida["constelacion_tx"] = constelacion_tx
        salida["constelacion_rx"] = np.concatenate(simbolos_rx_datos)
        if simbolos_rx_antes:                         # Solo en MRC (n_rx > 1): RX de 1 antena (antes)
            salida["constelacion_rx_antes"] = np.concatenate(simbolos_rx_antes)
    return salida


# =====================================================================================
# ===          2) SOPORTE — MODULACIONES QAM, PARÁMETROS, CANAL, RUIDO, IMAGEN       ===
# =====================================================================================
# Maquinaria de apoyo: no es el foco de la práctica, pero la cadena de arriba la usa.

# --- Modulaciones digitales QAM (Gray, energía media unitaria) ---

def qpsk_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a símbolos QPSK normalizados (energía media unitaria)."""
    bits = bits.reshape(-1, 2).astype(np.int8)
    i = 1 - 2 * bits[:, 0]
    q = 1 - 2 * bits[:, 1]
    return (i + 1j * q) / np.sqrt(2)


def qpsk_demapear(simbolos: np.ndarray) -> np.ndarray:
    """Demapeo por decisión dura (hard-decision) de QPSK."""
    s = simbolos * np.sqrt(2)
    b0 = (np.real(s) < 0).astype(np.uint8)
    b1 = (np.imag(s) < 0).astype(np.uint8)
    out = np.empty(2 * len(simbolos), dtype=np.uint8)
    out[0::2] = b0
    out[1::2] = b1
    return out


_GRAY_4 = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
_GRAY_4_INV = {v: k for k, v in _GRAY_4.items()}


def qam16_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a 16-QAM con codificación Gray, normalizado (E_s = 1)."""
    bits = bits.reshape(-1, 4)
    i = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 0:2]])
    q = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 2:4]])
    return (i + 1j * q) / np.sqrt(10)


def _demap_4pam_gray(x: np.ndarray) -> np.ndarray:
    """Demapeo hard-decision de un eje 4-PAM Gray a 2 bits por muestra."""
    niveles = np.array([-3, -1, 1, 3])
    idx = np.argmin(np.abs(x[:, None] - niveles[None, :]), axis=1)
    nivel = niveles[idx]
    bits = np.empty((len(x), 2), dtype=np.uint8)
    for k, n in enumerate(nivel):
        b0, b1 = _GRAY_4_INV[int(n)]
        bits[k, 0] = b0
        bits[k, 1] = b1
    return bits


def qam16_demapear(simbolos: np.ndarray) -> np.ndarray:
    """Demapeo hard-decision de 16-QAM."""
    s = simbolos * np.sqrt(10)
    bi = _demap_4pam_gray(np.real(s))
    bq = _demap_4pam_gray(np.imag(s))
    out = np.empty(4 * len(simbolos), dtype=np.uint8)
    out[0::4] = bi[:, 0]
    out[1::4] = bi[:, 1]
    out[2::4] = bq[:, 0]
    out[3::4] = bq[:, 1]
    return out


_GRAY_8 = {
    (0, 0, 0): -7, (0, 0, 1): -5, (0, 1, 1): -3, (0, 1, 0): -1,
    (1, 1, 0):  1, (1, 1, 1):  3, (1, 0, 1):  5, (1, 0, 0):  7,
}
_GRAY_8_INV = {v: k for k, v in _GRAY_8.items()}


def qam64_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a 64-QAM con codificación Gray, normalizado (E_s = 1)."""
    bits = bits.reshape(-1, 6)
    i = np.array([_GRAY_8[(int(a), int(b), int(c))] for a, b, c in bits[:, 0:3]])
    q = np.array([_GRAY_8[(int(a), int(b), int(c))] for a, b, c in bits[:, 3:6]])
    return (i + 1j * q) / np.sqrt(42)


def _demap_8pam_gray(x: np.ndarray) -> np.ndarray:
    """Demapeo hard-decision de un eje 8-PAM Gray a 3 bits por muestra."""
    niveles = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    idx = np.argmin(np.abs(x[:, None] - niveles[None, :]), axis=1)
    nivel = niveles[idx]
    bits = np.empty((len(x), 3), dtype=np.uint8)
    for k, n in enumerate(nivel):
        b0, b1, b2 = _GRAY_8_INV[int(n)]
        bits[k, 0] = b0
        bits[k, 1] = b1
        bits[k, 2] = b2
    return bits


def qam64_demapear(simbolos: np.ndarray) -> np.ndarray:
    """Demapeo hard-decision de 64-QAM."""
    s = simbolos * np.sqrt(42)
    bi = _demap_8pam_gray(np.real(s))
    bq = _demap_8pam_gray(np.imag(s))
    out = np.empty(6 * len(simbolos), dtype=np.uint8)
    out[0::6] = bi[:, 0]
    out[1::6] = bi[:, 1]
    out[2::6] = bi[:, 2]
    out[3::6] = bq[:, 0]
    out[4::6] = bq[:, 1]
    out[5::6] = bq[:, 2]
    return out


MODULACIONES = {
    "QPSK":   {"bits": 2, "mapear": qpsk_mapear,  "demapear": qpsk_demapear},
    "16-QAM": {"bits": 4, "mapear": qam16_mapear, "demapear": qam16_demapear},
    "64-QAM": {"bits": 6, "mapear": qam64_mapear, "demapear": qam64_demapear},
}

# El transmisor de esta práctica es OFDM (admite las 3 modulaciones)
MODS_OFDM = ["QPSK", "16-QAM", "64-QAM"]


# --- Parámetros de la numerología LTE ---

TABLA_NSC = {
    (1.4, 15): 72,   (3.0, 15): 180,  (5.0, 15): 300,  (10.0, 15): 600,
    (15.0, 15): 900, (20.0, 15): 1200, (5.0, 7.5): 600, (10.0, 7.5): 1200,
    (15.0, 7.5): 1800, (20.0, 7.5): 2400,
}


def siguiente_pot_2(n: int) -> int:
    """Menor potencia de 2 mayor o igual que n (para el tamaño de FFT)."""
    return 1 << (int(n - 1).bit_length())


def duracion_cp_us(delta_f_khz: float, tipo_cp: str) -> float:
    """Duración del prefijo cíclico en microsegundos según la numerología de LTE."""
    if delta_f_khz == 15:
        if tipo_cp == "normal":
            return 4.7
        if tipo_cp == "extendido":
            return 16.67
    if delta_f_khz == 7.5 and tipo_cp == "extendido":
        return 33.33
    raise ValueError("Combinación Δf/CP no válida")


def calcular_parametros_ofdm(bw_mhz: float, delta_f_khz: float, tipo_cp: str,
                              n_bits: int, bits_por_simbolo: int) -> Dict:
    """Traduce la configuración (BW, Δf, CP) en los parámetros físicos de la cadena OFDM."""
    clave = (round(bw_mhz, 1), float(delta_f_khz))
    if clave not in TABLA_NSC:
        raise ValueError(f"Combinación BW={bw_mhz} MHz / Δf={delta_f_khz} kHz no disponible")

    n_sc = TABLA_NSC[clave]
    n_fft = siguiente_pot_2(n_sc)
    fs = n_fft * delta_f_khz * 1e3
    dur_cp = duracion_cp_us(delta_f_khz, tipo_cp) * 1e-6
    n_cp = int(round(dur_cp * fs))

    n_pilotos = n_sc // PASO_PILOTO
    n_datos = n_sc - n_pilotos

    n_simbolos_qam = math.ceil(n_bits / bits_por_simbolo) if bits_por_simbolo > 0 else 0
    n_simbolos_ofdm = math.ceil(n_simbolos_qam / n_datos) if n_datos > 0 else 0
    duracion_simbolo_us = (n_fft + n_cp) / fs * 1e6

    return {
        "n_sc": int(n_sc), "n_fft": int(n_fft), "fs": float(fs), "n_cp": int(n_cp),
        "n_pilotos": int(n_pilotos), "n_datos": int(n_datos),
        "n_simbolos_qam": int(n_simbolos_qam), "n_simbolos_ofdm": int(n_simbolos_ofdm),
        "duracion_simbolo_us": float(duracion_simbolo_us), "duracion_cp_us": float(dur_cp * 1e6),
        "delta_f_hz": float(delta_f_khz * 1e3), "bw_hz": float(bw_mhz * 1e6),
    }


# --- Canal Pedestrian A, ruido AWGN y BER ---

RETARDOS_PEDA_NS = np.array([0.0, 110.0, 190.0, 410.0])
POTENCIAS_PEDA_DB = np.array([0.0, -9.7, -19.2, -22.8])


def generar_canal_pedestrian_a(n_fft: int, fs: float, indices_fft: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    """Respuesta en frecuencia H[k] del canal Pedestrian A (Rayleigh por tap)."""
    pot_lin = 10 ** (POTENCIAS_PEDA_DB / 10)
    pot_lin = pot_lin / np.sum(pot_lin)
    a = (rng.standard_normal(len(pot_lin)) + 1j * rng.standard_normal(len(pot_lin))) \
        * np.sqrt(pot_lin / 2)
    retardos_muestras = np.round(RETARDOS_PEDA_NS * 1e-9 * fs).astype(int)
    h = np.zeros(n_fft, dtype=complex)
    for tap, m in enumerate(retardos_muestras):
        if m < n_fft:
            h[m] += a[tap]
    H = np.fft.fft(h)
    return H[indices_fft]


def agregar_ruido_awgn(senal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Agrega ruido blanco gaussiano complejo (AWGN) calibrado al SNR indicado (en dB)."""
    pot_senal = np.mean(np.abs(senal) ** 2)
    pot_ruido = pot_senal / (10 ** (snr_db / 10))
    ruido = np.sqrt(pot_ruido / 2) * (rng.standard_normal(senal.shape) +
                                       1j * rng.standard_normal(senal.shape))
    return senal + ruido


def calcular_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    """Tasa de error de bit (BER) = bits distintos / bits comparados."""
    n = min(len(bits_tx), len(bits_rx))
    if n == 0:
        return 0.0
    return float(np.sum(bits_tx[:n] != bits_rx[:n]) / n)


# --- Imagen <-> bits ---

def imagen_a_bits(img: Image.Image) -> Tuple[np.ndarray, str, Tuple[int, int]]:
    """Convierte una imagen PIL en un vector de bits uint8 (MSB primero)."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    bits = np.unpackbits(arr.flatten())
    return bits, img.mode, img.size


def bits_a_imagen(bits: np.ndarray, mode: str, size: Tuple[int, int]) -> Image.Image:
    """Reconstruye una imagen PIL a partir del vector de bits recibido."""
    w, h = size
    canales = 3 if mode == "RGB" else 1
    n_bytes = w * h * canales
    n_bits = n_bytes * 8
    if len(bits) < n_bits:
        bits = np.concatenate([bits, np.zeros(n_bits - len(bits), dtype=np.uint8)])
    else:
        bits = bits[:n_bits]
    arr = np.packbits(bits).reshape((h, w, canales)) if canales == 3 else \
          np.packbits(bits).reshape((h, w))
    return Image.fromarray(arr, mode=mode)


def img_a_b64(img: Image.Image, formato: str = "PNG") -> str:
    """Serializa una imagen PIL a un data URI base64 para enviarla al navegador."""
    buf = io.BytesIO()
    img.save(buf, format=formato)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# =====================================================================================
# ===                3) MONTE CARLO + ENDPOINTS FLASK (no es el foco)                ===
# =====================================================================================
# Curvas BER vs SNR por número de antenas (Monte Carlo) y el servidor web. No es lo central.

app = Flask(__name__)
CARPETA_UPLOADS = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(CARPETA_UPLOADS, exist_ok=True)
app.config["UPLOAD_FOLDER"] = CARPETA_UPLOADS

ESTADO = {"imagen_bytes": None, "imagen_mode": None, "imagen_size": None, "bits": None, "formato": "PNG"}


@app.route("/")
def index():
    """Sirve la página principal."""
    return render_template("index.html")


@app.route("/subir_imagen", methods=["POST"])
def subir_imagen():
    """Recibe una imagen, la convierte en bits y la guarda en ESTADO."""
    if "imagen" not in request.files:
        return jsonify({"error": "No se envió imagen"}), 400
    f = request.files["imagen"]
    try:
        img = Image.open(f.stream)
        img.load()
    except Exception as e:
        return jsonify({"error": f"No se pudo leer la imagen: {e}"}), 400

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    bits, modo, size = imagen_a_bits(img)
    ESTADO["bits"] = bits
    ESTADO["imagen_mode"] = modo
    ESTADO["imagen_size"] = size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    ESTADO["imagen_bytes"] = buf.getvalue()

    return jsonify({
        "n_bits": int(len(bits)),
        "ancho": size[0],
        "alto": size[1],
        "canales": 3 if modo == "RGB" else 1,
        "preview_b64": "data:image/png;base64," + base64.b64encode(ESTADO["imagen_bytes"]).decode("ascii"),
    })


@app.route("/calcular_parametros", methods=["POST"])
def calcular_parametros():
    """Calcula y devuelve los parámetros derivados (N_SC, N_FFT, fs, N_CP, ...)."""
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        modulacion = data.get("modulacion", "16-QAM")
        bps = MODULACIONES[modulacion]["bits"]
        n_bits = int(data.get("n_bits", 0)) or (len(ESTADO["bits"]) if ESTADO["bits"] is not None else 0)
        params = calcular_parametros_ofdm(bw, df, cp, n_bits, bps)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(params)


@app.route("/simular_mrc", methods=["POST"])
def simular_mrc():
    """
    Ejecuta UNA transmisión de la imagen con transmisor OFDM y receptor de n_rx antenas (MRC).
    Devuelve la imagen recuperada, la BER y las constelaciones RX antes (1 antena, ZF) y
    después de MRC.
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        snr = float(data["snr_db"])
        modulacion = data["modulacion"]
        n_rx = int(data["n_rx"])
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    if modulacion not in MODS_OFDM:
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400
    if n_rx < 1:
        return jsonify({"error": "n_rx debe ser >= 1"}), 400
    if ESTADO["bits"] is None:
        return jsonify({"error": "Primero suba una imagen"}), 400

    bps = MODULACIONES[modulacion]["bits"]
    bits_tx = ESTADO["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, len(bits_tx), bps)

    rng = np.random.default_rng()
    t0 = time.perf_counter()
    resultado = cadena_tx_rx(bits_tx, modulacion, params, snr, rng,
                              capturar_constelaciones=True, n_rx=n_rx)
    tiempo_computo_s = time.perf_counter() - t0

    tiempo_aire_s = resultado["n_simbolos_ofdm"] * params["duracion_simbolo_us"] * 1e-6
    img_rx = bits_a_imagen(resultado["bits_rx"], ESTADO["imagen_mode"], ESTADO["imagen_size"])
    img_tx = Image.open(io.BytesIO(ESTADO["imagen_bytes"]))

    def submuestrear(arr, max_n=3000):
        if len(arr) > max_n:
            idx = np.random.choice(len(arr), max_n, replace=False)
            return arr[idx]
        return arr

    c_rx = submuestrear(resultado["constelacion_rx"])
    salida = {
        "ber": resultado["ber"],
        "modulacion": modulacion,
        "n_rx": n_rx,
        "snr_db": snr,
        "bits_transmitidos": int(len(bits_tx)),
        "bits_erroneos": int(round(resultado["ber"] * len(bits_tx))),
        "n_simbolos_ofdm": int(resultado["n_simbolos_ofdm"]),
        "imagen_original_b64": img_a_b64(img_tx),
        "imagen_recuperada_b64": img_a_b64(img_rx),
        "constelacion_rx": {"real": c_rx.real.tolist(), "imag": c_rx.imag.tolist()},
        "parametros": params,
        "tiempo_aire_s": tiempo_aire_s,
        "tiempo_computo_s": tiempo_computo_s,
    }
    if "constelacion_rx_antes" in resultado:                        # Constelación de 1 antena (antes de MRC)
        c_antes = submuestrear(resultado["constelacion_rx_antes"])
        salida["constelacion_rx_antes"] = {"real": c_antes.real.tolist(), "imag": c_antes.imag.tolist()}
    return jsonify(salida)


@app.route("/montecarlo_mrc", methods=["POST"])
def montecarlo_mrc():
    """
    Monte Carlo de BER vs SNR para diversidad en RX con MRC. Para la modulación seleccionada
    calcula una curva por cada nº de antenas en `antenas` (def. [2, 4, 8]). Al aumentar el nº
    de antenas la curva debe bajar (ganancia de diversidad). IC 95% con la t de Student.
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        modulacion = data["modulacion"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    if modulacion not in MODS_OFDM:
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400

    antenas = [int(a) for a in data.get("antenas", [2, 4, 8]) if int(a) >= 1]
    if not antenas:
        antenas = [2, 4, 8]

    N_BITS_MC = 10000                                  # Bits por corrida (reducido por la carga de MRC)
    snr_valores = list(range(0, 16))                   # Barrido de SNR de 0 a 15 dB
    n_sim = 8                                          # Corridas independientes por punto
    rng = np.random.default_rng()
    t_critico = float(t_student.ppf(0.975, df=n_sim - 1))

    bps = MODULACIONES[modulacion]["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)

    series_ber = []
    for n_rx in antenas:                               # Una curva BER vs SNR por cada nº de antenas
        ber_prom, ic_inf, ic_sup = [], [], []
        for snr in snr_valores:
            bers = []
            for _ in range(n_sim):
                bits_aleatorios = rng.integers(0, 2, N_BITS_MC, dtype=np.uint8)
                r = cadena_tx_rx(bits_aleatorios, modulacion, params, snr, rng,
                                  capturar_constelaciones=False, n_rx=n_rx)
                bers.append(r["ber"])
            bers = np.array(bers)
            mu = float(np.mean(bers))
            sd = float(np.std(bers, ddof=1)) if n_sim > 1 else 0.0
            margen = t_critico * sd / np.sqrt(n_sim)
            ber_prom.append(mu)
            ic_inf.append(max(mu - margen, 1e-6))
            ic_sup.append(mu + margen)
        series_ber.append({
            "n_rx": n_rx,
            "ber_promedio": ber_prom, "ic_inferior": ic_inf, "ic_superior": ic_sup,
        })

    return jsonify({
        "snr_valores": snr_valores,
        "n_simulaciones": n_sim,
        "t_critico": t_critico,
        "n_bits_mc": N_BITS_MC,
        "modulacion": modulacion,
        "series_ber": series_ber,
    })


# Punto de entrada: arranca el servidor de desarrollo Flask en el puerto 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
