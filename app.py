"""
Simulador OFDM-LTE — Flask
Toda la lógica de modulación, OFDM, canal Pedestrian A, AWGN, BER y Monte Carlo
está en este archivo. Comentarios y variables en español.
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

# === CONFIGURACIÓN FLASK ===
app = Flask(__name__)
CARPETA_UPLOADS = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(CARPETA_UPLOADS, exist_ok=True)
app.config["UPLOAD_FOLDER"] = CARPETA_UPLOADS

# Estado en memoria de la última imagen subida (mono-usuario, demo)
ESTADO = {
    "imagen_bytes": None,   # bytes originales (para devolver preview)
    "imagen_mode": None,    # modo PIL ('RGB' o 'L')
    "imagen_size": None,    # (w, h)
    "bits": None,           # np.ndarray uint8 con los bits aplanados
    "formato": "PNG",
}

# === TABLAS LTE ===
TABLA_NSC = {
    # (BW_MHz, delta_f_kHz) -> N_SC
    (1.4, 15): 72,
    (3.0, 15): 180,
    (5.0, 15): 300,
    (10.0, 15): 600,
    (15.0, 15): 900,
    (20.0, 15): 1200,
    (5.0, 7.5): 600,
    (10.0, 7.5): 1200,
    (15.0, 7.5): 1800,
    (20.0, 7.5): 2400,
}

# Canal Pedestrian A (ITU-R M.1225)
RETARDOS_PEDA_NS = np.array([0.0, 110.0, 190.0, 410.0])
POTENCIAS_PEDA_DB = np.array([0.0, -9.7, -19.2, -22.8])

PASO_PILOTO = 12  # 1 piloto cada 12 subportadoras


# =====================================================================
# === FUNCIONES DE MODULACIÓN (QPSK, 16-QAM, 64-QAM con Gray coding) ===
# =====================================================================

def qpsk_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a símbolos QPSK normalizados (energía media unitaria)."""
    bits = bits.reshape(-1, 2).astype(np.int8)
    # Gray: 0 -> +1, 1 -> -1
    i = 1 - 2 * bits[:, 0]
    q = 1 - 2 * bits[:, 1]
    return (i + 1j * q) / np.sqrt(2)


def qpsk_demapear(simbolos: np.ndarray) -> np.ndarray:
    """Demapeo hard-decision QPSK."""
    s = simbolos * np.sqrt(2)
    b0 = (np.real(s) < 0).astype(np.uint8)
    b1 = (np.imag(s) < 0).astype(np.uint8)
    out = np.empty(2 * len(simbolos), dtype=np.uint8)
    out[0::2] = b0
    out[1::2] = b1
    return out


# Mapa Gray para 4-PAM (16-QAM): 2 bits -> nivel {-3,-1,+1,+3}
_GRAY_4 = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
_GRAY_4_INV = {v: k for k, v in _GRAY_4.items()}


def qam16_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a 16-QAM Gray, normalizado (E_s = 1)."""
    bits = bits.reshape(-1, 4)
    i = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 0:2]])
    q = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 2:4]])
    return (i + 1j * q) / np.sqrt(10)


def _demap_4pam_gray(x: np.ndarray) -> np.ndarray:
    """Demapeo hard-decision de 4-PAM Gray a 2 bits por muestra."""
    # Decidir nivel más cercano en {-3,-1,1,3}
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
    """Demapeo hard-decision 16-QAM."""
    s = simbolos * np.sqrt(10)
    bi = _demap_4pam_gray(np.real(s))
    bq = _demap_4pam_gray(np.imag(s))
    out = np.empty(4 * len(simbolos), dtype=np.uint8)
    out[0::4] = bi[:, 0]
    out[1::4] = bi[:, 1]
    out[2::4] = bq[:, 0]
    out[3::4] = bq[:, 1]
    return out


# Mapa Gray para 8-PAM (64-QAM): 3 bits -> nivel {-7,-5,-3,-1,+1,+3,+5,+7}
_GRAY_8 = {
    (0, 0, 0): -7, (0, 0, 1): -5, (0, 1, 1): -3, (0, 1, 0): -1,
    (1, 1, 0):  1, (1, 1, 1):  3, (1, 0, 1):  5, (1, 0, 0):  7,
}
_GRAY_8_INV = {v: k for k, v in _GRAY_8.items()}


def qam64_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a 64-QAM Gray, normalizado (E_s = 1)."""
    bits = bits.reshape(-1, 6)
    i = np.array([_GRAY_8[(int(a), int(b), int(c))] for a, b, c in bits[:, 0:3]])
    q = np.array([_GRAY_8[(int(a), int(b), int(c))] for a, b, c in bits[:, 3:6]])
    return (i + 1j * q) / np.sqrt(42)


def _demap_8pam_gray(x: np.ndarray) -> np.ndarray:
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
    """Demapeo hard-decision 64-QAM."""
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


# Diccionario por modulación
MODULACIONES = {
    "QPSK":   {"bits": 2, "mapear": qpsk_mapear,  "demapear": qpsk_demapear},
    "16-QAM": {"bits": 4, "mapear": qam16_mapear, "demapear": qam16_demapear},
    "64-QAM": {"bits": 6, "mapear": qam64_mapear, "demapear": qam64_demapear},
}

# Esquemas de acceso múltiple.
#  - OFDM   : sin precodificación (Práctica 1), 3 modulaciones.
#  - SC-FDM : DFT-spread antes de la IFFT (Práctica 2, uplink LTE). Solo QPSK y 16-QAM.
ESQUEMAS = {
    "OFDM":   {"dft": False, "mods": ["QPSK", "16-QAM", "64-QAM"]},
    "SC-FDM": {"dft": True,  "mods": ["QPSK", "16-QAM"]},
}


# =====================================================================
# === FUNCIONES OFDM                                                 ===
# =====================================================================

def siguiente_pot_2(n: int) -> int:
    """Siguiente potencia de 2 >= n."""
    return 1 << (int(n - 1).bit_length())


# Tyrone check
def duracion_cp_us(delta_f_khz: float, tipo_cp: str) -> float:
    """Duración del prefijo cíclico en microsegundos según LTE."""
    if delta_f_khz == 15:
        if tipo_cp == "normal":
            return 4.7
        if tipo_cp == "extendido":
            return 16.67
    if delta_f_khz == 7.5 and tipo_cp == "extendido":
        return 33.33
    raise ValueError("Combinación Δf/CP no válida")

# Tyrone
def calcular_parametros_ofdm(bw_mhz: float, delta_f_khz: float, tipo_cp: str,
                              n_bits: int, bits_por_simbolo: int) -> Dict:
    """Calcula todos los parámetros derivados de la configuración OFDM."""
    clave = (round(bw_mhz, 1), float(delta_f_khz))
    if clave not in TABLA_NSC:
        raise ValueError(f"Combinación BW={bw_mhz} MHz / Δf={delta_f_khz} kHz no disponible")
    
    n_sc = TABLA_NSC[clave]
    n_fft = siguiente_pot_2(n_sc)
    fs = n_fft * delta_f_khz * 1e3  # Hz
    dur_cp = duracion_cp_us(delta_f_khz, tipo_cp) * 1e-6
    n_cp = int(round(dur_cp * fs))

    n_pilotos = n_sc // PASO_PILOTO
    n_datos = n_sc - n_pilotos

    n_simbolos_qam = math.ceil(n_bits / bits_por_simbolo) if bits_por_simbolo > 0 else 0
    n_simbolos_ofdm = math.ceil(n_simbolos_qam / n_datos) if n_datos > 0 else 0

    duracion_simbolo_us = (n_fft + n_cp) / fs * 1e6

    return {
        "n_sc": int(n_sc),
        "n_fft": int(n_fft),
        "fs": float(fs),
        "n_cp": int(n_cp),
        "n_pilotos": int(n_pilotos),
        "n_datos": int(n_datos),
        "n_simbolos_qam": int(n_simbolos_qam),
        "n_simbolos_ofdm": int(n_simbolos_ofdm),
        "duracion_simbolo_us": float(duracion_simbolo_us),
        "duracion_cp_us": float(dur_cp * 1e6),
        "delta_f_hz": float(delta_f_khz * 1e3),
        "bw_hz": float(bw_mhz * 1e6),
    }

# Tyrone
def indices_pilotos_datos(n_sc: int, paso: int = PASO_PILOTO) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve los índices (dentro de [0, n_sc-1]) de pilotos y datos."""
    indices = np.arange(n_sc)
    pilotos = indices[::paso]
    datos = np.setdiff1d(indices, pilotos, assume_unique=False)
    return pilotos, datos

# Tyrone
def insertar_pilotos(simbolos_datos: np.ndarray, n_sc: int,
                     paso: int = PASO_PILOTO, valor_piloto: complex = 1 + 0j) -> np.ndarray:
    """Construye un vector de longitud n_sc con pilotos cada `paso` posiciones."""
    pilotos, datos = indices_pilotos_datos(n_sc, paso)
    rejilla = np.zeros(n_sc, dtype=complex)
    rejilla[pilotos] = valor_piloto
    # Tomar solo los símbolos de datos que entran en esta rejilla
    n = min(len(datos), len(simbolos_datos))
    rejilla[datos[:n]] = simbolos_datos[:n]
    return rejilla

# Tyrone
def extraer_datos_de_pilotos(simbolos_con_pilotos: np.ndarray, n_sc: int,
                              paso: int = PASO_PILOTO) -> np.ndarray:
    """Extrae solo los símbolos de datos (descarta posiciones piloto)."""
    _, datos = indices_pilotos_datos(n_sc, paso)
    return simbolos_con_pilotos[datos]

# Freddy
def mapeo_sc_a_fft(rejilla_sc: np.ndarray, n_fft: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mapea las N_SC subportadoras al vector de N_FFT puntos en banda base,
    dejando DC vacío y centrando los índices [-N_SC/2, ..., -1, +1, ..., +N_SC/2].
    Retorna (vector_freq, indices_fft_usados).
    """
    n_sc = len(rejilla_sc)
    mitad = n_sc // 2
    # Índices físicos en el vector de FFT (orden numpy: 0..N/2-1 son freq positivas, N/2..N-1 son negativas)
    indices_pos = np.arange(1, mitad + 1)             # +1..+mitad
    indices_neg = n_fft - np.arange(1, mitad + 1)[::-1]  # equivalentes a -mitad..-1
    indices_fft = np.concatenate([indices_neg, indices_pos])
    vec = np.zeros(n_fft, dtype=complex)
    vec[indices_fft] = rejilla_sc
    return vec, indices_fft

# Freddy
def modulacion_ofdm(rejilla_sc: np.ndarray, n_fft: int, n_cp: int) -> Tuple[np.ndarray, np.ndarray]:
    """IFFT + CP. Retorna (señal_en_tiempo, indices_fft_activos)."""
    vec_freq, indices_fft = mapeo_sc_a_fft(rejilla_sc, n_fft)
    senal_t = np.fft.ifft(vec_freq) * n_fft / np.sqrt(len(rejilla_sc))  # normalización
    cp = senal_t[-n_cp:] if n_cp > 0 else np.array([], dtype=complex)
    return np.concatenate([cp, senal_t]), indices_fft

# Freddy
def demodulacion_ofdm(simbolo_con_cp: np.ndarray, indices_fft: np.ndarray,
                       n_fft: int, n_cp: int, n_sc: int) -> np.ndarray:
    """Remueve CP, aplica FFT y extrae las N_SC subportadoras activas."""
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)
    return espectro[indices_fft]

# Freddy
def generar_canal_pedestrian_a(n_fft: int, fs: float, indices_fft: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    """
    Genera la respuesta en frecuencia H[k] del canal Pedestrian A
    (Rayleigh fading por tap) sobre las subportadoras activas.
    """
    pot_lin = 10 ** (POTENCIAS_PEDA_DB / 10)
    pot_lin = pot_lin / np.sum(pot_lin)  # normalizar potencia total a 1
    # Coeficientes complejos gaussianos por tap
    a = (rng.standard_normal(len(pot_lin)) + 1j * rng.standard_normal(len(pot_lin))) \
        * np.sqrt(pot_lin / 2)
    # Respuesta impulsiva discreta: muestra los taps en la rejilla temporal de fs
    retardos_muestras = np.round(RETARDOS_PEDA_NS * 1e-9 * fs).astype(int)
    h = np.zeros(n_fft, dtype=complex)
    for tap, m in enumerate(retardos_muestras):
        if m < n_fft:
            h[m] += a[tap]
    H = np.fft.fft(h)
    return H[indices_fft]

# Tyrone
def agregar_ruido_awgn(senal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Agrega AWGN complejo según el SNR dado (en dB)."""
    pot_senal = np.mean(np.abs(senal) ** 2)
    pot_ruido = pot_senal / (10 ** (snr_db / 10))
    ruido = np.sqrt(pot_ruido / 2) * (rng.standard_normal(senal.shape) +
                                       1j * rng.standard_normal(senal.shape))
    return senal + ruido


def calcular_papr_simbolo(senal_tiempo: np.ndarray) -> float:
    """PAPR (en dB) de un símbolo OFDM en tiempo (sin CP)."""
    pot_pico = np.max(np.abs(senal_tiempo) ** 2)
    pot_media = np.mean(np.abs(senal_tiempo) ** 2)
    if pot_media <= 0:
        return 0.0
    return float(10 * np.log10(pot_pico / pot_media))


def calcular_papr_ccdf(modulacion: str, params: Dict,
                        n_simbolos: int, rng: np.random.Generator,
                        usar_dft_spread: bool = False,
                        eje_x: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera n_simbolos (OFDM o SC-FDM) aleatorios para la modulación y devuelve
    (eje_x_dB, ccdf) donde ccdf[i] = Pr{PAPR > x_db[i]}.
    Si se pasa eje_x se usa ese eje común (necesario para superponer curvas).
    """
    mod = MODULACIONES[modulacion]
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    n_datos = params["n_datos"]
    bps = mod["bits"]

    paprs = np.empty(n_simbolos)
    for i in range(n_simbolos):
        bits = rng.integers(0, 2, n_datos * bps, dtype=np.uint8)
        simbolos_qam = mod["mapear"](bits)
        # OFDM (datos+pilotos) o SC-FDM (DFT-spread en bloque contiguo localizado)
        rejilla = construir_rejilla_tx(simbolos_qam, n_sc, n_datos, usar_dft_spread)
        senal_t, _ = modulacion_ofdm(rejilla, n_fft, n_cp)
        # quitar CP para PAPR sobre el símbolo puro
        paprs[i] = calcular_papr_simbolo(senal_t[n_cp:])

    if eje_x is None:
        eje_x = np.linspace(0, max(15.0, float(np.max(paprs)) + 0.5), 80)
    ccdf = np.array([np.mean(paprs > x) for x in eje_x])
    return eje_x, ccdf


def calcular_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    """Bit Error Rate entre dos vectores."""
    n = min(len(bits_tx), len(bits_rx))
    if n == 0:
        return 0.0
    return float(np.sum(bits_tx[:n] != bits_rx[:n]) / n)


# =====================================================================
# === SC-FDM: PRECODIFICACIÓN DFT-SPREAD                            ===
# =====================================================================
# SC-FDM (SC-FDMA, uplink de LTE) = OFDM + una DFT de tamaño M aplicada al
# bloque de símbolos de datos ANTES del mapeo a subportadoras y la IFFT.
# En el receptor se aplica la IDFT (despread) tras la ecualización. Esto
# reduce notablemente el PAPR manteniendo las mismas prestaciones de BER.

def dft_spread(bloque_datos: np.ndarray) -> np.ndarray:
    """DFT-spread unitaria de tamaño M = len(bloque_datos) (precoding SC-FDM)."""
    m = len(bloque_datos)
    if m == 0:
        return bloque_datos
    # Normalización unitaria: conserva la energía por subportadora (E_s = 1),
    # de modo que el BER de SC-FDM sea comparable al de OFDM.
    return np.fft.fft(bloque_datos) / np.sqrt(m)


def idft_despread(bloque_freq: np.ndarray) -> np.ndarray:
    """IDFT-despread unitaria (inversa exacta de dft_spread) en el receptor SC-FDM."""
    m = len(bloque_freq)
    if m == 0:
        return bloque_freq
    return np.fft.ifft(bloque_freq) * np.sqrt(m)


def construir_rejilla_tx(bloque_datos: np.ndarray, n_sc: int, n_datos: int,
                          usar_dft_spread: bool) -> np.ndarray:
    """
    Construye la rejilla de n_sc subportadoras del TX según el esquema:
      - OFDM   : símbolos QAM en las posiciones de datos + pilotos intercalados (cada 12).
      - SC-FDM : DFT-spread de los datos colocado en un BLOQUE CONTIGUO (mapeo localizado).
                 No se intercalan pilotos constantes en el símbolo de datos: eso preserva la
                 propiedad de portadora única y por tanto el bajo PAPR (en LTE las señales de
                 referencia DMRS van en símbolos dedicados, justamente por esta razón).
    """
    if usar_dft_spread:
        bloque_spread = dft_spread(np.asarray(bloque_datos, dtype=complex)[:n_datos])
        rejilla = np.zeros(n_sc, dtype=complex)
        rejilla[:len(bloque_spread)] = bloque_spread
        return rejilla
    return insertar_pilotos(bloque_datos, n_sc)


def forma_onda_simbolo(simbolos_qam_bloque: np.ndarray, params: Dict,
                        usar_dft_spread: bool, max_pts: int = 512) -> Tuple[list, float]:
    """
    Genera UN símbolo (OFDM o SC-FDM) a partir de un bloque de símbolos QAM y
    devuelve (envolvente |x(t)| submuestreada, PAPR en dB). Sirve para comparar
    visualmente la forma de onda en el tiempo de OFDM vs SC-FDM con los mismos datos.
    """
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    n_datos = params["n_datos"]

    bloque = np.asarray(simbolos_qam_bloque, dtype=complex)[:n_datos]
    if len(bloque) < n_datos:
        bloque = np.concatenate([bloque, np.zeros(n_datos - len(bloque), dtype=complex)])
    rejilla = construir_rejilla_tx(bloque, n_sc, n_datos, usar_dft_spread)
    senal_t, _ = modulacion_ofdm(rejilla, n_fft, n_cp)
    senal = senal_t[n_cp:]  # símbolo puro, sin CP
    papr = calcular_papr_simbolo(senal)
    envolvente = np.abs(senal)
    if len(envolvente) > max_pts:
        idx = np.linspace(0, len(envolvente) - 1, max_pts).astype(int)
        envolvente = envolvente[idx]
    return envolvente.tolist(), float(papr)


# =====================================================================
# === CADENA TX/RX COMPLETA                                          ===
# =====================================================================

# Freddy
def cadena_tx_rx(bits_tx: np.ndarray, modulacion: str, params: Dict,
                  snr_db: float, rng: np.random.Generator,
                  capturar_constelaciones: bool = False,
                  usar_dft_spread: bool = False) -> Dict:
    """
    Ejecuta TX -> Canal Pedestrian A -> AWGN -> RX.
    Con usar_dft_spread=True la cadena es SC-FDM (DFT-spread + IDFT-despread);
    con False es OFDM clásico.
    Retorna dict con bits_rx, ber y (opcional) constelaciones tx/rx.
    """
    mod = MODULACIONES[modulacion]
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    n_datos = params["n_datos"]
    fs = params["fs"]

    # === TX ===
    # Padding de bits para múltiplo de bits/símbolo
    bps = mod["bits"]
    falta = (-len(bits_tx)) % bps
    if falta:
        bits_tx_pad = np.concatenate([bits_tx, np.zeros(falta, dtype=np.uint8)])
    else:
        bits_tx_pad = bits_tx
    simbolos_qam = mod["mapear"](bits_tx_pad)

    # Padding de símbolos para llenar un número entero de símbolos OFDM
    falta_sim = (-len(simbolos_qam)) % n_datos
    if falta_sim:
        simbolos_qam = np.concatenate([simbolos_qam, np.zeros(falta_sim, dtype=complex)])
    n_ofdm = len(simbolos_qam) // n_datos

    constelacion_tx = simbolos_qam.copy() if capturar_constelaciones else None
    simbolos_rx_datos = []

    bits_rx_total = []
    _, indices_datos_sc = indices_pilotos_datos(n_sc)
    for i in range(n_ofdm):
        bloque_datos = simbolos_qam[i * n_datos:(i + 1) * n_datos]
        # TX: OFDM (datos+pilotos) o SC-FDM (DFT-spread en bloque contiguo localizado)
        rejilla = construir_rejilla_tx(bloque_datos, n_sc, n_datos, usar_dft_spread)
        senal_t, indices_fft = modulacion_ofdm(rejilla, n_fft, n_cp)

        # === Canal Pedestrian A: aplicamos en frecuencia ===
        H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)
        rejilla_freq = rejilla * H
        senal_canal, _ = modulacion_ofdm(rejilla_freq, n_fft, n_cp)

        # === AWGN ===
        senal_rx = agregar_ruido_awgn(senal_canal, snr_db, rng)

        # === RX con ecualización Zero-Forcing (canal conocido) ===
        rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
        rejilla_eq = rejilla_rx / H
        if usar_dft_spread:
            # SC-FDM: extraer el bloque contiguo de datos e IDFT-despread
            datos_rx = idft_despread(rejilla_eq[:n_datos])
        else:
            datos_rx = rejilla_eq[indices_datos_sc]
        if capturar_constelaciones:
            simbolos_rx_datos.append(datos_rx)

        bits_rx_total.append(mod["demapear"](datos_rx))

    bits_rx = np.concatenate(bits_rx_total) if bits_rx_total else np.array([], dtype=np.uint8)
    # Recortar al tamaño original transmitido (antes de padding de bits)
    bits_rx = bits_rx[:len(bits_tx_pad)]
    ber = calcular_ber(bits_tx_pad, bits_rx)

    salida = {
        "bits_rx": bits_rx[:len(bits_tx)],  # quitar también el padding inicial
        "ber": ber,
        "n_simbolos_ofdm": n_ofdm,
    }
    if capturar_constelaciones:
        salida["constelacion_tx"] = constelacion_tx
        salida["constelacion_rx"] = np.concatenate(simbolos_rx_datos)
    return salida


# =====================================================================
# === UTILIDADES IMAGEN <-> BITS                                     ===
# =====================================================================

def imagen_a_bits(img: Image.Image) -> Tuple[np.ndarray, str, Tuple[int, int]]:
    """Convierte una imagen PIL a un vector de bits uint8 (MSB primero)."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    bits = np.unpackbits(arr.flatten())
    return bits, img.mode, img.size


def bits_a_imagen(bits: np.ndarray, mode: str, size: Tuple[int, int]) -> Image.Image:
    """Reconstruye una imagen PIL a partir del vector de bits."""
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
    buf = io.BytesIO()
    img.save(buf, format=formato)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# =====================================================================
# === ENDPOINTS FLASK                                                ===
# =====================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/subir_imagen", methods=["POST"])
def subir_imagen():
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


@app.route("/simular", methods=["POST"])
def simular():
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        snr = float(data["snr_db"])
        modulacion = data["modulacion"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    esquema = data.get("esquema", "OFDM")
    if esquema not in ESQUEMAS:
        return jsonify({"error": f"Esquema no válido: {esquema}"}), 400
    if modulacion not in ESQUEMAS[esquema]["mods"]:
        return jsonify({"error": f"{modulacion} no está disponible en {esquema}"}), 400
    usar_dft = ESQUEMAS[esquema]["dft"]

    if ESTADO["bits"] is None:
        return jsonify({"error": "Primero suba una imagen"}), 400

    bps = MODULACIONES[modulacion]["bits"]
    bits_tx = ESTADO["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, len(bits_tx), bps)

    rng = np.random.default_rng()
    t0 = time.perf_counter()
    resultado = cadena_tx_rx(bits_tx, modulacion, params, snr, rng,
                              capturar_constelaciones=True, usar_dft_spread=usar_dft)
    tiempo_computo_s = time.perf_counter() - t0

    # Comparación de forma de onda |x(t)|: un símbolo representativo generado con datos
    # ALEATORIOS (los MISMOS para ambos esquemas) para evidenciar el menor PAPR de SC-FDM.
    # Se usan datos aleatorios (no el bloque de imagen, que es muy estructurado y daría un
    # PAPR atípico) de forma coherente con la metodología de la CCDF del PAPR.
    n_bits_bloque = params["n_datos"] * bps
    bits_bloque = rng.integers(0, 2, n_bits_bloque, dtype=np.uint8)
    simbolos_bloque = MODULACIONES[modulacion]["mapear"](bits_bloque)
    env_ofdm, papr_ofdm = forma_onda_simbolo(simbolos_bloque, params, usar_dft_spread=False)
    env_scfdm, papr_scfdm = forma_onda_simbolo(simbolos_bloque, params, usar_dft_spread=True)

    # Tiempo "real" de transmisión OTA: N_OFDM × duración símbolo
    tiempo_aire_s = resultado["n_simbolos_ofdm"] * params["duracion_simbolo_us"] * 1e-6

    # Reconstruir imagen
    img_rx = bits_a_imagen(resultado["bits_rx"], ESTADO["imagen_mode"], ESTADO["imagen_size"])
    img_tx = Image.open(io.BytesIO(ESTADO["imagen_bytes"]))

    # Submuestrear constelaciones para no enviar megabytes
    def submuestrear(arr, max_n=3000):
        if len(arr) > max_n:
            idx = np.random.choice(len(arr), max_n, replace=False)
            return arr[idx]
        return arr

    c_tx = submuestrear(resultado["constelacion_tx"])
    c_rx = submuestrear(resultado["constelacion_rx"])

    # Mapa de subportadoras según el esquema:
    #  - OFDM   : datos + pilotos intercalados (cada 12).
    #  - SC-FDM : bloque CONTIGUO de datos (localizado) + subportadoras de guarda; sin pilotos.
    if usar_dft:
        datos_idx = np.arange(params["n_datos"])
        pilotos_idx = np.array([], dtype=int)
        guarda_idx = np.arange(params["n_datos"], params["n_sc"])
    else:
        pilotos_idx, datos_idx = indices_pilotos_datos(params["n_sc"])
        guarda_idx = np.array([], dtype=int)

    return jsonify({
        "ber": resultado["ber"],
        "esquema": esquema,
        "bits_transmitidos": int(len(bits_tx)),
        "bits_erroneos": int(round(resultado["ber"] * len(bits_tx))),
        "modulacion": modulacion,
        "n_simbolos_ofdm": int(resultado["n_simbolos_ofdm"]),
        "snr_db": snr,
        "imagen_original_b64": img_a_b64(img_tx),
        "imagen_recuperada_b64": img_a_b64(img_rx),
        "constelacion_tx": {"real": c_tx.real.tolist(), "imag": c_tx.imag.tolist()},
        "constelacion_rx": {"real": c_rx.real.tolist(), "imag": c_rx.imag.tolist()},
        "mapa_subportadoras": {
            "total": int(params["n_sc"]),
            "pilotos": int(len(pilotos_idx)),
            "datos": int(len(datos_idx)),
            "guarda": int(len(guarda_idx)),
            "tipo_mapeo": "localizado" if usar_dft else "distribuido",
            "indices_pilotos": pilotos_idx.tolist(),
            "indices_datos": datos_idx.tolist(),
            "indices_guarda": guarda_idx.tolist(),
        },
        "forma_onda": {
            "ofdm": {"envolvente": env_ofdm, "papr_db": papr_ofdm},
            "scfdm": {"envolvente": env_scfdm, "papr_db": papr_scfdm},
        },
        "parametros": params,
        "tiempo_aire_s": tiempo_aire_s,
        "tiempo_computo_s": tiempo_computo_s,
        "throughput_bps": (len(bits_tx) / tiempo_aire_s) if tiempo_aire_s > 0 else 0.0,
    })

# Tyrone
@app.route("/montecarlo", methods=["POST"])
def montecarlo():
    """
    Simulación Monte Carlo. Acepta `esquemas` (lista). Devuelve series planas:
      - esquemas=["OFDM"]            -> 3 curvas (Práctica 1, pestaña OFDM)
      - esquemas=["OFDM","SC-FDM"]   -> 5 curvas combinadas (Práctica 2, pestaña SC-FDM)
    Cada serie lleva su esquema y modulación para que el frontend la pinte
    (color=modulación, estilo=esquema).
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    # Esquemas a simular (validados contra ESQUEMAS)
    esquemas = [e for e in data.get("esquemas", ["OFDM"]) if e in ESQUEMAS]
    if not esquemas:
        esquemas = ["OFDM"]
    combinado = len(esquemas) > 1

    # En modo combinado se reduce la carga (5×16×10 corridas) para que sea ágil
    N_BITS_MC = 20000 if combinado else 50000
    N_SIMBOLOS_PAPR = 1000 if combinado else 1500
    snr_valores = list(range(0, 16))
    n_sim = 10
    rng = np.random.default_rng()

    # Valor crítico T-Student 95%, n-1 = 9 gl
    t_critico = float(t_student.ppf(0.975, df=n_sim - 1))

    # === BER vs SNR (una serie por esquema×modulación) ===
    series_ber = []
    for esquema in esquemas:
        usar_dft = ESQUEMAS[esquema]["dft"]
        for modulacion in ESQUEMAS[esquema]["mods"]:
            bps = MODULACIONES[modulacion]["bits"]
            params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)
            ber_prom, ic_inf, ic_sup = [], [], []
            for snr in snr_valores:
                bers = []
                for _ in range(n_sim):
                    bits_aleatorios = rng.integers(0, 2, N_BITS_MC, dtype=np.uint8)
                    r = cadena_tx_rx(bits_aleatorios, modulacion, params, snr, rng,
                                      capturar_constelaciones=False, usar_dft_spread=usar_dft)
                    bers.append(r["ber"])
                bers = np.array(bers)
                mu = float(np.mean(bers))
                sd = float(np.std(bers, ddof=1)) if n_sim > 1 else 0.0
                margen = t_critico * sd / np.sqrt(n_sim)
                ber_prom.append(mu)
                ic_inf.append(max(mu - margen, 1e-6))
                ic_sup.append(mu + margen)
            series_ber.append({
                "esquema": esquema,
                "modulacion": modulacion,
                "ber_promedio": ber_prom,
                "ic_inferior": ic_inf,
                "ic_superior": ic_sup,
            })

    # === CCDF del PAPR (independiente del SNR), sobre un eje común ===
    eje_comun = np.linspace(0, 15, 80)
    series_papr = []
    for esquema in esquemas:
        usar_dft = ESQUEMAS[esquema]["dft"]
        for modulacion in ESQUEMAS[esquema]["mods"]:
            bps = MODULACIONES[modulacion]["bits"]
            params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)
            _, c = calcular_papr_ccdf(modulacion, params, N_SIMBOLOS_PAPR, rng,
                                       usar_dft_spread=usar_dft, eje_x=eje_comun)
            series_papr.append({
                "esquema": esquema,
                "modulacion": modulacion,
                "ccdf": c.tolist(),
            })

    return jsonify({
        "snr_valores": snr_valores,
        "n_simulaciones": n_sim,
        "t_critico": t_critico,
        "n_bits_mc": N_BITS_MC,
        "series_ber": series_ber,
        "papr": {
            "x_db": eje_comun.tolist(),
            "series": series_papr,
            "n_simbolos": N_SIMBOLOS_PAPR,
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
