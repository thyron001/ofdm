"""
==========================================================================================
 Simulador OFDM / SC-FDM conforme a LTE — Backend Flask
==========================================================================================
 Toda la lógica de procesamiento de señal (modulación QAM, OFDM, SC-FDM con DFT-spread,
 canal Pedestrian A, AWGN, ecualización, BER, PAPR y Monte Carlo) vive en este único
 archivo. El código está SEGMENTADO en bloques para que sea fácil distinguir qué pertenece
 a cada práctica y qué es compartido:

   [BLOQUE 0]  Configuración Flask y estado en memoria.
   [BLOQUE 1]  COMÚN  — Constantes (tablas LTE, canal Pedestrian A) y esquemas.
   [BLOQUE 2]  COMÚN  — Modulaciones digitales QAM (las usan OFDM y SC-FDM por igual).
   [BLOQUE 3]  COMÚN  — Parámetros, canal, ruido, BER, PAPR de un símbolo e imagen<->bits.
   [BLOQUE 4]  OFDM   — Núcleo OFDM: pilotos, mapeo de subportadoras, IFFT/FFT + CP.
   [BLOQUE 5]  SC-FDM — Precodificación DFT-spread / IDFT-despread y mapeo localizado.
   [BLOQUE 6]  CADENA UNIFICADA — TX->Canal->RX que se ramifica a OFDM o SC-FDM, PAPR-CCDF.
   [BLOQUE 7]  ENDPOINTS Flask — exponen la lógica a la interfaz web.

 Convención: con "usar_dft_spread=False" la cadena es OFDM (Práctica 1) y con
 "usar_dft_spread=True" es SC-FDM (Práctica 2). Comentarios y variables en español.
==========================================================================================
"""

# =====================================================================================
# === [BLOQUE 0] IMPORTS Y CONFIGURACIÓN FLASK                                       ===
# =====================================================================================

import os                                   # Rutas y creación de carpetas del sistema
import io                                    # Buffers en memoria para imágenes (BytesIO)
import time                                  # Medición de tiempos de cómputo
import base64                               # Codificar imágenes a texto (data URI)
import math                                  # Funciones matemáticas escalares (ceil)
from typing import Dict, Tuple              # Anotaciones de tipo para legibilidad

import numpy as np                           # Cálculo numérico vectorizado (FFT, arrays)
from scipy.stats import t as t_student       # Distribución t de Student (intervalos de confianza)
from PIL import Image                        # Lectura/escritura de imágenes
from flask import Flask, render_template, request, jsonify  # Servidor web y JSON

# Instancia principal de la aplicación Flask
app = Flask(__name__)
# Carpeta donde se guardarían las imágenes subidas (se crea si no existe)
CARPETA_UPLOADS = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(CARPETA_UPLOADS, exist_ok=True)
app.config["UPLOAD_FOLDER"] = CARPETA_UPLOADS

# Estado en memoria de la última imagen subida (aplicación mono-usuario, de demostración).
# Lo comparten ambas pestañas (OFDM y SC-FDM): la última imagen cargada sirve a las dos.
ESTADO = {
    "imagen_bytes": None,   # bytes originales de la imagen (para devolver la vista previa)
    "imagen_mode": None,    # modo de color PIL ('RGB' o 'L')
    "imagen_size": None,    # dimensiones (ancho, alto)
    "bits": None,           # np.ndarray uint8 con todos los bits de la imagen aplanados
    "formato": "PNG",       # formato de salida usado al reconstruir
}


# =====================================================================================
# === [BLOQUE 1] CONSTANTES COMUNES (TABLAS LTE, CANAL) Y ESQUEMAS                   ===
# =====================================================================================

# Tabla de la numerología LTE: para cada par (ancho de banda en MHz, Δf en kHz) indica el
# número de subportadoras útiles N_SC. La comparten OFDM y SC-FDM.
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

# Perfil potencia-retardo del canal Pedestrian A (ITU-R M.1225): 4 trayectorias (taps).
RETARDOS_PEDA_NS = np.array([0.0, 110.0, 190.0, 410.0])     # Retardos de cada tap en ns
POTENCIAS_PEDA_DB = np.array([0.0, -9.7, -19.2, -22.8])     # Potencia relativa de cada tap en dB

PASO_PILOTO = 12  # En OFDM se inserta 1 subportadora piloto cada 12 subportadoras

# Esquemas de acceso múltiple soportados por el simulador:
#   - OFDM   : sin precodificación (Práctica 1). Admite las 3 modulaciones.
#   - SC-FDM : DFT-spread antes de la IFFT (Práctica 2, enlace ascendente de LTE). Solo 2.
# El campo "dft" indica si se aplica la precodificación DFT-spread; "mods" lista las
# modulaciones válidas de cada esquema.
ESQUEMAS = {
    "OFDM":   {"dft": False, "mods": ["QPSK", "16-QAM", "64-QAM"]},
    "SC-FDM": {"dft": True,  "mods": ["QPSK", "16-QAM"]},
}


# =====================================================================================
# === [BLOQUE 2] COMÚN — MODULACIONES DIGITALES QAM (Gray, energía media unitaria)   ===
# =====================================================================================
# Estas funciones convierten bits <-> símbolos del plano I/Q. Son IDÉNTICAS para OFDM y
# SC-FDM: ambos esquemas modulan los bits igual; la diferencia entre prácticas aparece
# después, en cómo se mapean esos símbolos a las subportadoras.

def qpsk_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a símbolos QPSK normalizados (energía media unitaria)."""
    bits = bits.reshape(-1, 2).astype(np.int8)   # Agrupa los bits de dos en dos (2 bits/símbolo)
    # Codificación Gray por eje: bit 0 -> +1, bit 1 -> -1
    i = 1 - 2 * bits[:, 0]                        # Componente en fase (I) a partir del primer bit
    q = 1 - 2 * bits[:, 1]                        # Componente en cuadratura (Q) a partir del segundo bit
    return (i + 1j * q) / np.sqrt(2)             # Combina I+jQ y normaliza a energía unitaria (/√2)


def qpsk_demapear(simbolos: np.ndarray) -> np.ndarray:
    """Demapeo por decisión dura (hard-decision) de QPSK."""
    s = simbolos * np.sqrt(2)                     # Deshace la normalización para volver a {±1}
    b0 = (np.real(s) < 0).astype(np.uint8)       # Bit 0 = signo de la parte real (negativa -> 1)
    b1 = (np.imag(s) < 0).astype(np.uint8)       # Bit 1 = signo de la parte imaginaria
    out = np.empty(2 * len(simbolos), dtype=np.uint8)  # Vector de bits de salida
    out[0::2] = b0                                # Coloca los bits 0 en posiciones pares
    out[1::2] = b1                                # Coloca los bits 1 en posiciones impares
    return out


# Mapa Gray para 4-PAM (eje de 16-QAM): 2 bits -> nivel de amplitud {-3,-1,+1,+3}
_GRAY_4 = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
_GRAY_4_INV = {v: k for k, v in _GRAY_4.items()}  # Mapa inverso nivel -> par de bits (para demapear)


def qam16_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a 16-QAM con codificación Gray, normalizado (E_s = 1)."""
    bits = bits.reshape(-1, 4)                                            # 4 bits por símbolo
    i = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 0:2]])   # Eje I: primeros 2 bits -> 4-PAM
    q = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 2:4]])   # Eje Q: últimos 2 bits -> 4-PAM
    return (i + 1j * q) / np.sqrt(10)                                    # Normaliza por √10 (energía media 16-QAM)


def _demap_4pam_gray(x: np.ndarray) -> np.ndarray:
    """Demapeo hard-decision de un eje 4-PAM Gray a 2 bits por muestra."""
    niveles = np.array([-3, -1, 1, 3])                              # Niveles válidos del eje
    idx = np.argmin(np.abs(x[:, None] - niveles[None, :]), axis=1)  # Índice del nivel más cercano
    nivel = niveles[idx]                                            # Nivel decidido por mínima distancia
    bits = np.empty((len(x), 2), dtype=np.uint8)                   # 2 bits por muestra
    for k, n in enumerate(nivel):                                  # Recorre cada nivel decidido
        b0, b1 = _GRAY_4_INV[int(n)]                               # Recupera los 2 bits del mapa inverso
        bits[k, 0] = b0
        bits[k, 1] = b1
    return bits


def qam16_demapear(simbolos: np.ndarray) -> np.ndarray:
    """Demapeo hard-decision de 16-QAM."""
    s = simbolos * np.sqrt(10)                       # Deshace la normalización
    bi = _demap_4pam_gray(np.real(s))                # Demapea el eje en fase (I)
    bq = _demap_4pam_gray(np.imag(s))                # Demapea el eje en cuadratura (Q)
    out = np.empty(4 * len(simbolos), dtype=np.uint8)  # 4 bits por símbolo
    out[0::4] = bi[:, 0]                             # Intercala los bits I y Q en el orden de mapeo
    out[1::4] = bi[:, 1]
    out[2::4] = bq[:, 0]
    out[3::4] = bq[:, 1]
    return out


# Mapa Gray para 8-PAM (eje de 64-QAM): 3 bits -> nivel {-7,-5,-3,-1,+1,+3,+5,+7}
_GRAY_8 = {
    (0, 0, 0): -7, (0, 0, 1): -5, (0, 1, 1): -3, (0, 1, 0): -1,
    (1, 1, 0):  1, (1, 1, 1):  3, (1, 0, 1):  5, (1, 0, 0):  7,
}
_GRAY_8_INV = {v: k for k, v in _GRAY_8.items()}  # Mapa inverso nivel -> 3 bits


def qam64_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a 64-QAM con codificación Gray, normalizado (E_s = 1)."""
    bits = bits.reshape(-1, 6)                                                       # 6 bits por símbolo
    i = np.array([_GRAY_8[(int(a), int(b), int(c))] for a, b, c in bits[:, 0:3]])    # Eje I: 3 bits -> 8-PAM
    q = np.array([_GRAY_8[(int(a), int(b), int(c))] for a, b, c in bits[:, 3:6]])    # Eje Q: 3 bits -> 8-PAM
    return (i + 1j * q) / np.sqrt(42)                                                # Normaliza por √42 (64-QAM)


def _demap_8pam_gray(x: np.ndarray) -> np.ndarray:
    """Demapeo hard-decision de un eje 8-PAM Gray a 3 bits por muestra."""
    niveles = np.array([-7, -5, -3, -1, 1, 3, 5, 7])               # Niveles válidos del eje
    idx = np.argmin(np.abs(x[:, None] - niveles[None, :]), axis=1)  # Nivel más cercano
    nivel = niveles[idx]
    bits = np.empty((len(x), 3), dtype=np.uint8)                   # 3 bits por muestra
    for k, n in enumerate(nivel):
        b0, b1, b2 = _GRAY_8_INV[int(n)]                           # Recupera los 3 bits del mapa inverso
        bits[k, 0] = b0
        bits[k, 1] = b1
        bits[k, 2] = b2
    return bits


def qam64_demapear(simbolos: np.ndarray) -> np.ndarray:
    """Demapeo hard-decision de 64-QAM."""
    s = simbolos * np.sqrt(42)                       # Deshace la normalización
    bi = _demap_8pam_gray(np.real(s))                # Demapea el eje I
    bq = _demap_8pam_gray(np.imag(s))                # Demapea el eje Q
    out = np.empty(6 * len(simbolos), dtype=np.uint8)  # 6 bits por símbolo
    out[0::6] = bi[:, 0]
    out[1::6] = bi[:, 1]
    out[2::6] = bi[:, 2]
    out[3::6] = bq[:, 0]
    out[4::6] = bq[:, 1]
    out[5::6] = bq[:, 2]
    return out


# Diccionario que asocia cada modulación con sus bits/símbolo y sus funciones de mapeo.
# Lo consultan tanto la cadena OFDM como la SC-FDM.
MODULACIONES = {
    "QPSK":   {"bits": 2, "mapear": qpsk_mapear,  "demapear": qpsk_demapear},
    "16-QAM": {"bits": 4, "mapear": qam16_mapear, "demapear": qam16_demapear},
    "64-QAM": {"bits": 6, "mapear": qam64_mapear, "demapear": qam64_demapear},
}


# =====================================================================================
# === [BLOQUE 3] COMÚN — PARÁMETROS, CANAL, RUIDO, BER, PAPR DE UN SÍMBOLO, IMAGEN   ===
# =====================================================================================
# Toda esta lógica es independiente del esquema: se aplica igual a OFDM y a SC-FDM.

def siguiente_pot_2(n: int) -> int:
    """Devuelve la menor potencia de 2 mayor o igual que n (para el tamaño de FFT)."""
    return 1 << (int(n - 1).bit_length())


def duracion_cp_us(delta_f_khz: float, tipo_cp: str) -> float:
    """Duración del prefijo cíclico en microsegundos según la numerología de LTE."""
    if delta_f_khz == 15:                       # Con Δf = 15 kHz hay CP normal y extendido
        if tipo_cp == "normal":
            return 4.7
        if tipo_cp == "extendido":
            return 16.67
    if delta_f_khz == 7.5 and tipo_cp == "extendido":  # Con Δf = 7.5 kHz solo hay CP extendido
        return 33.33
    raise ValueError("Combinación Δf/CP no válida")


def calcular_parametros_ofdm(bw_mhz: float, delta_f_khz: float, tipo_cp: str,
                              n_bits: int, bits_por_simbolo: int) -> Dict:
    """
    Traduce la configuración de usuario (BW, Δf, tipo de CP) en los parámetros físicos
    de la cadena. Sirve por igual a OFDM y SC-FDM: en SC-FDM, n_datos hace además de
    tamaño M de la DFT-spread.
    """
    clave = (round(bw_mhz, 1), float(delta_f_khz))       # Clave de búsqueda en la tabla LTE
    if clave not in TABLA_NSC:                            # Valida que la combinación BW/Δf exista
        raise ValueError(f"Combinación BW={bw_mhz} MHz / Δf={delta_f_khz} kHz no disponible")

    n_sc = TABLA_NSC[clave]                               # Número de subportadoras útiles
    n_fft = siguiente_pot_2(n_sc)                         # Tamaño de FFT (potencia de 2 >= N_SC)
    fs = n_fft * delta_f_khz * 1e3                        # Frecuencia de muestreo fs = N_FFT·Δf (Hz)
    dur_cp = duracion_cp_us(delta_f_khz, tipo_cp) * 1e-6  # Duración del CP en segundos
    n_cp = int(round(dur_cp * fs))                        # Longitud del CP en muestras

    n_pilotos = n_sc // PASO_PILOTO                       # Subportadoras piloto (1 de cada 12)
    n_datos = n_sc - n_pilotos                            # Subportadoras de datos (también M en SC-FDM)

    # Número de símbolos QAM y de símbolos OFDM/SC-FDM necesarios para la carga útil
    n_simbolos_qam = math.ceil(n_bits / bits_por_simbolo) if bits_por_simbolo > 0 else 0
    n_simbolos_ofdm = math.ceil(n_simbolos_qam / n_datos) if n_datos > 0 else 0

    duracion_simbolo_us = (n_fft + n_cp) / fs * 1e6      # Duración total del símbolo (FFT+CP) en µs

    return {                                              # Diccionario de parámetros derivados
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


def generar_canal_pedestrian_a(n_fft: int, fs: float, indices_fft: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    """
    Genera la respuesta en frecuencia H[k] del canal Pedestrian A (desvanecimiento de
    Rayleigh por tap) sobre las subportadoras activas. Es el mismo canal para ambos esquemas.
    """
    pot_lin = 10 ** (POTENCIAS_PEDA_DB / 10)             # Pasa las potencias de los taps de dB a lineal
    pot_lin = pot_lin / np.sum(pot_lin)                  # Normaliza la potencia total del canal a 1
    # Coeficiente complejo gaussiano por tap (Rayleigh): real e imag con varianza P/2
    a = (rng.standard_normal(len(pot_lin)) + 1j * rng.standard_normal(len(pot_lin))) \
        * np.sqrt(pot_lin / 2)
    retardos_muestras = np.round(RETARDOS_PEDA_NS * 1e-9 * fs).astype(int)  # Retardos -> muestras
    h = np.zeros(n_fft, dtype=complex)                   # Respuesta al impulso discreta
    for tap, m in enumerate(retardos_muestras):          # Coloca cada tap en su posición temporal
        if m < n_fft:
            h[m] += a[tap]
    H = np.fft.fft(h)                                    # Respuesta en frecuencia = FFT de h
    return H[indices_fft]                               # Devuelve H solo en las subportadoras activas


def agregar_ruido_awgn(senal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Agrega ruido blanco gaussiano complejo (AWGN) calibrado al SNR indicado (en dB)."""
    pot_senal = np.mean(np.abs(senal) ** 2)             # Potencia media de la señal
    pot_ruido = pot_senal / (10 ** (snr_db / 10))       # Potencia de ruido para el SNR objetivo
    ruido = np.sqrt(pot_ruido / 2) * (rng.standard_normal(senal.shape) +
                                       1j * rng.standard_normal(senal.shape))  # Ruido complejo
    return senal + ruido                                # Señal contaminada


def _agregar_ruido(senal: np.ndarray, snr_db: float, rng: np.random.Generator,
                   pot_ref=None) -> np.ndarray:
    """
    AWGN con dos modos de calibración:
      - pot_ref None      -> calibra el ruido a la potencia RECIBIDA por realización
                             (agregar_ruido_awgn). Es el comportamiento de las Prácticas 1-4.
      - pot_ref = <float> -> PISO DE RUIDO FIJO referido a esa potencia de TRANSMISIÓN de
                             referencia (la de una antena, SISO). El piso NO depende de cuánta
                             señal llega, así la GANANCIA DE ARREGLO del beamforming (la señal RX
                             crece ×N_T) se refleja en la BER, y BF/SFBC/MRC quedan sobre el MISMO
                             piso para poder compararlas de forma justa (Práctica 5).
    """
    if pot_ref is None:
        return agregar_ruido_awgn(senal, snr_db, rng)
    pot_ruido = pot_ref / (10 ** (snr_db / 10))
    return senal + np.sqrt(pot_ruido / 2) * (rng.standard_normal(senal.shape) +
                                             1j * rng.standard_normal(senal.shape))


def calcular_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    """Tasa de error de bit (BER) = bits distintos / bits comparados. Común a ambos esquemas."""
    n = min(len(bits_tx), len(bits_rx))                 # Compara solo la parte común
    if n == 0:
        return 0.0
    return float(np.sum(bits_tx[:n] != bits_rx[:n]) / n)


def calcular_papr_simbolo(senal_tiempo: np.ndarray) -> float:
    """PAPR (en dB) de un símbolo en el dominio del tiempo (sin CP). Métrica común."""
    pot_pico = np.max(np.abs(senal_tiempo) ** 2)        # Potencia instantánea máxima (pico)
    pot_media = np.mean(np.abs(senal_tiempo) ** 2)      # Potencia media
    if pot_media <= 0:
        return 0.0
    return float(10 * np.log10(pot_pico / pot_media))   # PAPR en dB = 10·log10(pico/media)


def imagen_a_bits(img: Image.Image) -> Tuple[np.ndarray, str, Tuple[int, int]]:
    """Convierte una imagen PIL en un vector de bits uint8 (MSB primero)."""
    if img.mode not in ("RGB", "L"):                    # Solo se admiten color RGB o escala de grises
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)                 # Matriz de píxeles (0-255)
    bits = np.unpackbits(arr.flatten())                 # Desempaqueta cada byte en 8 bits
    return bits, img.mode, img.size


def bits_a_imagen(bits: np.ndarray, mode: str, size: Tuple[int, int]) -> Image.Image:
    """Reconstruye una imagen PIL a partir del vector de bits recibido."""
    w, h = size                                         # Ancho y alto
    canales = 3 if mode == "RGB" else 1                 # 3 canales (RGB) o 1 (gris)
    n_bytes = w * h * canales                           # Bytes totales esperados
    n_bits = n_bytes * 8                                # Bits totales esperados
    if len(bits) < n_bits:                              # Rellena con ceros si faltan bits
        bits = np.concatenate([bits, np.zeros(n_bits - len(bits), dtype=np.uint8)])
    else:                                               # Recorta si sobran
        bits = bits[:n_bits]
    arr = np.packbits(bits).reshape((h, w, canales)) if canales == 3 else \
          np.packbits(bits).reshape((h, w))            # Reagrupa bits en bytes y reordena la matriz
    return Image.fromarray(arr, mode=mode)             # Devuelve la imagen PIL reconstruida


def img_a_b64(img: Image.Image, formato: str = "PNG") -> str:
    """Serializa una imagen PIL a un data URI base64 para enviarla al navegador."""
    buf = io.BytesIO()                                  # Buffer en memoria
    img.save(buf, format=formato)                       # Guarda la imagen en el buffer
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# =====================================================================================
# === [BLOQUE 4] OFDM — NÚCLEO: PILOTOS, MAPEO DE SUBPORTADORAS, IFFT/FFT + CP        ===
# =====================================================================================
# Estas funciones implementan la forma de onda OFDM. SC-FDM REUTILIZA la IFFT/FFT+CP
# (modulacion_ofdm / demodulacion_ofdm), pero NO usa la inserción de pilotos intercalados:
# por eso esas dos rutinas se consideran el "núcleo OFDM" compartido y los pilotos quedan
# como parte específica de OFDM.

def indices_pilotos_datos(n_sc: int, paso: int = PASO_PILOTO) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve los índices (en [0, n_sc-1]) de las subportadoras piloto y de datos (OFDM)."""
    indices = np.arange(n_sc)                           # Todas las subportadoras
    pilotos = indices[::paso]                           # Pilotos: una de cada `paso` (12)
    datos = np.setdiff1d(indices, pilotos, assume_unique=False)  # El resto son datos
    return pilotos, datos


def insertar_pilotos(simbolos_datos: np.ndarray, n_sc: int,
                     paso: int = PASO_PILOTO, valor_piloto: complex = 1 + 0j) -> np.ndarray:
    """Construye la rejilla OFDM de n_sc subportadoras con pilotos intercalados cada `paso`."""
    pilotos, datos = indices_pilotos_datos(n_sc, paso)  # Posiciones de pilotos y de datos
    rejilla = np.zeros(n_sc, dtype=complex)             # Rejilla vacía
    rejilla[pilotos] = valor_piloto                     # Coloca el valor piloto conocido (1+0j)
    n = min(len(datos), len(simbolos_datos))            # Cuántos símbolos de datos caben
    rejilla[datos[:n]] = simbolos_datos[:n]             # Coloca los símbolos de datos en su sitio
    return rejilla


def extraer_datos_de_pilotos(simbolos_con_pilotos: np.ndarray, n_sc: int,
                              paso: int = PASO_PILOTO) -> np.ndarray:
    """Extrae solo las subportadoras de datos de una rejilla OFDM (descarta los pilotos)."""
    _, datos = indices_pilotos_datos(n_sc, paso)        # Posiciones de datos
    return simbolos_con_pilotos[datos]


def mapeo_sc_a_fft(rejilla_sc: np.ndarray, n_fft: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mapea las N_SC subportadoras al vector de N_FFT puntos en banda base, dejando la
    componente DC vacía y centrando los índices en [-N_SC/2,...,-1,+1,...,+N_SC/2].
    Lo usan tanto OFDM como SC-FDM (ambos pasan por la IFFT).
    Retorna (vector_frecuencia, índices_fft_usados).
    """
    n_sc = len(rejilla_sc)                              # Número de subportadoras activas
    mitad = n_sc // 2                                   # Mitad para repartir en frecuencias ±
    indices_pos = np.arange(1, mitad + 1)              # Índices de frecuencias positivas (+1..+mitad)
    indices_neg = n_fft - np.arange(1, mitad + 1)[::-1]  # Índices de frecuencias negativas (equiv. -mitad..-1)
    indices_fft = np.concatenate([indices_neg, indices_pos])  # Orden monótono en frecuencia
    vec = np.zeros(n_fft, dtype=complex)               # Vector espectral lleno de ceros (DC y guardas)
    vec[indices_fft] = rejilla_sc                      # Coloca las subportadoras activas
    return vec, indices_fft


def modulacion_ofdm(rejilla_sc: np.ndarray, n_fft: int, n_cp: int) -> Tuple[np.ndarray, np.ndarray]:
    """IFFT + prefijo cíclico. Genera la señal en el tiempo. La usan OFDM y SC-FDM."""
    vec_freq, indices_fft = mapeo_sc_a_fft(rejilla_sc, n_fft)        # Vector espectral centrado
    senal_t = np.fft.ifft(vec_freq) * n_fft / np.sqrt(len(rejilla_sc))  # IFFT con normalización de energía
    cp = senal_t[-n_cp:] if n_cp > 0 else np.array([], dtype=complex)   # Prefijo cíclico = últimas n_cp muestras
    return np.concatenate([cp, senal_t]), indices_fft               # Antepone el CP a la señal


def demodulacion_ofdm(simbolo_con_cp: np.ndarray, indices_fft: np.ndarray,
                       n_fft: int, n_cp: int, n_sc: int) -> np.ndarray:
    """Quita el CP, aplica la FFT y extrae las N_SC subportadoras activas. La usan ambos esquemas."""
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]                       # Descarta el prefijo cíclico
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)           # FFT con normalización inversa
    return espectro[indices_fft]                                   # Devuelve solo las subportadoras activas


# =====================================================================================
# === [BLOQUE 5] SC-FDM — PRECODIFICACIÓN DFT-SPREAD Y MAPEO LOCALIZADO              ===
# =====================================================================================
# Este es el bloque que DISTINGUE a SC-FDM de OFDM. SC-FDM (SC-FDMA, enlace ascendente de
# LTE) añade una DFT de tamaño M sobre el bloque de datos ANTES de la IFFT y la IDFT
# correspondiente en el receptor. El resultado se mapea a un BLOQUE CONTIGUO de
# subportadoras (mapeo localizado), lo que conserva la propiedad de portadora única y
# reduce notablemente el PAPR manteniendo el mismo BER que OFDM.

def dft_spread(bloque_datos: np.ndarray) -> np.ndarray:
    """DFT-spread unitaria de tamaño M = len(bloque_datos) (precodificación SC-FDM)."""
    m = len(bloque_datos)                               # Tamaño M de la DFT (= nº de símbolos de datos)
    if m == 0:
        return bloque_datos
    # Normalización unitaria (/√M): conserva la energía por subportadora (E_s = 1) para que
    # el BER de SC-FDM sea comparable al de OFDM.
    return np.fft.fft(bloque_datos) / np.sqrt(m)


def idft_despread(bloque_freq: np.ndarray) -> np.ndarray:
    """IDFT-despread unitaria (inversa exacta de dft_spread) en el receptor SC-FDM."""
    m = len(bloque_freq)                               # Tamaño M
    if m == 0:
        return bloque_freq
    return np.fft.ifft(bloque_freq) * np.sqrt(m)       # ·√M deshace la /√M del transmisor


def construir_rejilla_tx(bloque_datos: np.ndarray, n_sc: int, n_datos: int,
                          usar_dft_spread: bool) -> np.ndarray:
    """
    Construye la rejilla de n_sc subportadoras del transmisor según el esquema. ESTE es el
    punto donde la cadena se ramifica en OFDM o SC-FDM:
      - OFDM   : símbolos QAM en las posiciones de datos + pilotos intercalados (cada 12).
      - SC-FDM : DFT-spread de los datos colocado en un BLOQUE CONTIGUO (mapeo localizado).
                 No se intercalan pilotos constantes en el símbolo de datos: eso preserva la
                 propiedad de portadora única y por tanto el bajo PAPR (en LTE las señales de
                 referencia DMRS van en símbolos dedicados, justamente por esta razón).
    """
    if usar_dft_spread:                                                       # --- Rama SC-FDM ---
        bloque_spread = dft_spread(np.asarray(bloque_datos, dtype=complex)[:n_datos])  # DFT de M datos
        rejilla = np.zeros(n_sc, dtype=complex)                              # Rejilla vacía (resto = guarda)
        rejilla[:len(bloque_spread)] = bloque_spread                        # Coloca el bloque contiguo al inicio
        return rejilla
    return insertar_pilotos(bloque_datos, n_sc)                             # --- Rama OFDM: datos + pilotos ---


# =====================================================================================
# === [BLOQUE 6] CADENA UNIFICADA TX -> CANAL -> RX (OFDM o SC-FDM) + PAPR-CCDF      ===
# =====================================================================================
# Estas rutinas funcionan para AMBOS esquemas: el parámetro booleano "usar_dft_spread"
# selecciona OFDM (False) o SC-FDM (True). Internamente llaman a construir_rejilla_tx (que
# elige el mapeo) y, en SC-FDM, aplican la IDFT-despread tras la ecualización.

def forma_onda_simbolo(simbolos_qam_bloque: np.ndarray, params: Dict,
                        usar_dft_spread: bool, max_pts: int = 512) -> Tuple[list, float]:
    """
    Genera UN símbolo (OFDM o SC-FDM) a partir de un bloque de símbolos QAM y devuelve
    (envolvente |x(t)| submuestreada, PAPR en dB). Sirve para comparar visualmente la forma
    de onda en el tiempo de OFDM frente a SC-FDM usando los mismos datos.
    """
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    n_datos = params["n_datos"]

    bloque = np.asarray(simbolos_qam_bloque, dtype=complex)[:n_datos]   # Toma hasta n_datos símbolos
    if len(bloque) < n_datos:                                           # Rellena con ceros si faltan
        bloque = np.concatenate([bloque, np.zeros(n_datos - len(bloque), dtype=complex)])
    rejilla = construir_rejilla_tx(bloque, n_sc, n_datos, usar_dft_spread)  # Mapeo OFDM o SC-FDM
    senal_t, _ = modulacion_ofdm(rejilla, n_fft, n_cp)                 # IFFT + CP
    senal = senal_t[n_cp:]                                             # Símbolo puro, sin CP, para el PAPR
    papr = calcular_papr_simbolo(senal)                               # PAPR en dB de este símbolo
    envolvente = np.abs(senal)                                        # Envolvente |x(t)|
    if len(envolvente) > max_pts:                                     # Submuestrea para no enviar miles de puntos
        idx = np.linspace(0, len(envolvente) - 1, max_pts).astype(int)
        envolvente = envolvente[idx]
    return envolvente.tolist(), float(papr)


def calcular_papr_ccdf(modulacion: str, params: Dict,
                        n_simbolos: int, rng: np.random.Generator,
                        usar_dft_spread: bool = False,
                        eje_x: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estima la CCDF del PAPR generando n_simbolos aleatorios (OFDM o SC-FDM según el flag).
    Devuelve (eje_x_dB, ccdf) con ccdf[i] = Pr{PAPR > x_db[i]}. Si se pasa eje_x se usa ese
    eje común (imprescindible para superponer las curvas de OFDM y SC-FDM en una gráfica).
    """
    mod = MODULACIONES[modulacion]
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    n_datos = params["n_datos"]
    bps = mod["bits"]

    paprs = np.empty(n_simbolos)                                       # Vector con el PAPR de cada símbolo
    for i in range(n_simbolos):                                        # Genera muchos símbolos aleatorios
        bits = rng.integers(0, 2, n_datos * bps, dtype=np.uint8)      # Bits aleatorios de un símbolo
        simbolos_qam = mod["mapear"](bits)                           # Mapea a símbolos QAM
        # Construye la rejilla con el mismo mapeo que el TX real (OFDM o SC-FDM)
        rejilla = construir_rejilla_tx(simbolos_qam, n_sc, n_datos, usar_dft_spread)
        senal_t, _ = modulacion_ofdm(rejilla, n_fft, n_cp)           # IFFT + CP
        paprs[i] = calcular_papr_simbolo(senal_t[n_cp:])            # PAPR del símbolo (sin CP)

    if eje_x is None:                                                 # Si no hay eje común, se crea uno
        eje_x = np.linspace(0, max(15.0, float(np.max(paprs)) + 0.5), 80)
    ccdf = np.array([np.mean(paprs > x) for x in eje_x])             # CCDF: fracción que supera cada umbral
    return eje_x, ccdf


def cadena_tx_rx(bits_tx: np.ndarray, modulacion: str, params: Dict,
                  snr_db: float, rng: np.random.Generator,
                  capturar_constelaciones: bool = False,
                  usar_dft_spread: bool = False, n_rx: int = 1,
                  ruido_piso_fijo: bool = False) -> Dict:
    """
    Núcleo del simulador: ejecuta TX -> canal Pedestrian A -> AWGN -> RX para un vector de
    bits. Con usar_dft_spread=False es OFDM; con True es SC-FDM (añade DFT-spread/IDFT).
    Devuelve un dict con los bits recibidos, la BER y, opcionalmente, las constelaciones.

    n_rx = nº de antenas en el receptor (Práctica 3, diversidad en RX):
      - n_rx == 1  -> receptor clásico de 1 antena con ecualización Zero-Forcing (Y/H).
      - n_rx  > 1  -> el MISMO símbolo TX llega por n_rx canales independientes, cada uno con
                      su propio ruido, y se combinan con MRC (Maximal Ratio Combining):
                      X_est[k] = Σ_m conj(H_m[k])·Y_m[k] / Σ_m |H_m[k]|².
                      Al aumentar n_rx mejora la SNR efectiva y la BER baja (diversidad).
    """
    mod = MODULACIONES[modulacion]
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    n_datos = params["n_datos"]
    fs = params["fs"]

    # --- TX: bits -> símbolos QAM (igual en ambos esquemas) ---
    bps = mod["bits"]                                  # Bits por símbolo de la modulación
    falta = (-len(bits_tx)) % bps                      # Bits que faltan para múltiplo de bps
    if falta:                                          # Rellena con ceros si no es múltiplo
        bits_tx_pad = np.concatenate([bits_tx, np.zeros(falta, dtype=np.uint8)])
    else:
        bits_tx_pad = bits_tx
    simbolos_qam = mod["mapear"](bits_tx_pad)         # Mapea todos los bits a símbolos QAM

    # Rellena los símbolos hasta completar un número entero de símbolos OFDM/SC-FDM
    falta_sim = (-len(simbolos_qam)) % n_datos
    if falta_sim:
        simbolos_qam = np.concatenate([simbolos_qam, np.zeros(falta_sim, dtype=complex)])
    n_ofdm = len(simbolos_qam) // n_datos             # Nº de símbolos OFDM/SC-FDM a transmitir

    constelacion_tx = simbolos_qam.copy() if capturar_constelaciones else None  # Constelación ideal TX
    simbolos_rx_datos = []                            # Acumulador de símbolos RX combinados (constelación)
    simbolos_rx_antes = []                            # Acumulador de símbolos RX de 1 antena (antes de MRC)

    bits_rx_total = []                                # Acumulador de bits recibidos
    _, indices_datos_sc = indices_pilotos_datos(n_sc)  # Posiciones de datos (se usan solo en OFDM)
    for i in range(n_ofdm):                            # Procesa símbolo a símbolo
        bloque_datos = simbolos_qam[i * n_datos:(i + 1) * n_datos]  # Bloque de n_datos símbolos
        # TX: rejilla OFDM (datos+pilotos) o SC-FDM (DFT-spread en bloque contiguo localizado)
        rejilla = construir_rejilla_tx(bloque_datos, n_sc, n_datos, usar_dft_spread)
        senal_tx_ref, indices_fft = modulacion_ofdm(rejilla, n_fft, n_cp)  # IFFT + CP (indices y, si aplica, ref)
        # Potencia de TX de referencia (1 antena) para el piso de ruido fijo de la Práctica 5;
        # None -> calibración clásica al recibido (comportamiento de las Prácticas 1-3).
        pot_ref = float(np.mean(np.abs(senal_tx_ref) ** 2)) if ruido_piso_fijo else None

        if n_rx <= 1:
            # --- Receptor de 1 antena: canal Pedestrian A + ecualización Zero-Forcing ---
            H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)  # Respuesta del canal H[k]
            senal_canal, _ = modulacion_ofdm(rejilla * H, n_fft, n_cp)   # Y[k]=H[k]·X[k], vuelve al tiempo
            senal_rx = _agregar_ruido(senal_canal, snr_db, rng, pot_ref)  # AWGN al SNR objetivo
            rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)  # Quita CP + FFT
            rejilla_eq = rejilla_rx / H                                  # Ecualización ZF: divide por H[k]
        else:
            # --- Diversidad en RX con MRC: n_rx antenas, cada una con canal y ruido propios ---
            num = np.zeros(n_sc, dtype=complex)         # Numerador MRC: Σ conj(H_m)·Y_m
            den = np.zeros(n_sc, dtype=float)           # Denominador MRC: Σ |H_m|²
            rejilla_eq_antes = None                     # Estimación con 1 sola antena (referencia "antes")
            for m in range(n_rx):
                H_m = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)  # Canal independiente por antena
                senal_canal, _ = modulacion_ofdm(rejilla * H_m, n_fft, n_cp)   # Y_m[k]=H_m[k]·X[k], al tiempo
                senal_rx = _agregar_ruido(senal_canal, snr_db, rng, pot_ref)   # Ruido independiente por antena
                rejilla_rx_m = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
                num += np.conj(H_m) * rejilla_rx_m       # Pondera y alinea en fase (combinación coherente)
                den += np.abs(H_m) ** 2                  # Suma de ganancias de canal
                if m == 0:
                    rejilla_eq_antes = rejilla_rx_m / H_m  # ZF de la 1ª antena (sin diversidad)
            rejilla_eq = num / (den + 1e-12)            # Estimación MRC de X[k]
            if capturar_constelaciones and not usar_dft_spread:
                simbolos_rx_antes.append(rejilla_eq_antes[indices_datos_sc])

        if usar_dft_spread:                                          # --- Rama SC-FDM ---
            # Extrae el bloque contiguo de datos y aplica la IDFT-despread
            datos_rx = idft_despread(rejilla_eq[:n_datos])
        else:                                                        # --- Rama OFDM ---
            datos_rx = rejilla_eq[indices_datos_sc]                  # Toma las posiciones de datos
        if capturar_constelaciones:                                  # Guarda símbolos RX si se pidió
            simbolos_rx_datos.append(datos_rx)

        bits_rx_total.append(mod["demapear"](datos_rx))             # Demapea a bits y acumula

    bits_rx = np.concatenate(bits_rx_total) if bits_rx_total else np.array([], dtype=np.uint8)
    bits_rx = bits_rx[:len(bits_tx_pad)]              # Recorta al tamaño transmitido (sin padding de símbolos)
    ber = calcular_ber(bits_tx_pad, bits_rx)          # Calcula la BER

    salida = {                                        # Resultado base
        "bits_rx": bits_rx[:len(bits_tx)],            # Bits útiles (sin el padding inicial de bits)
        "ber": ber,
        "n_simbolos_ofdm": n_ofdm,
    }
    if capturar_constelaciones:                       # Adjunta las constelaciones si se solicitaron
        salida["constelacion_tx"] = constelacion_tx
        salida["constelacion_rx"] = np.concatenate(simbolos_rx_datos)
        if simbolos_rx_antes:                         # Solo en MRC (n_rx > 1): RX de 1 antena (antes)
            salida["constelacion_rx_antes"] = np.concatenate(simbolos_rx_antes)
    return salida


def cadena_tx_sfbc_rx(bits_tx: np.ndarray, modulacion: str, params: Dict,
                       snr_db: float, rng: np.random.Generator,
                       capturar_constelaciones: bool = False,
                       n_tx: int = 2, n_rx: int = 1,
                       ruido_piso_fijo: bool = False) -> Dict:
    """
    Núcleo de la Práctica 4 (diversidad en TX con SFBC). Es la hermana de cadena_tx_rx pero el
    transmisor usa n_tx antenas y un mapeo por PARES de subportadoras adyacentes:

      - n_tx == 1  -> SISO de referencia (sin diversidad): canal Pedestrian A + ecualización ZF
                      (Y/H). Si n_rx > 1 se combinan las antenas RX con MRC. Diversidad = n_rx.
      - n_tx == 2  -> SFBC (código de Alamouti espacio-frecuencia): cada par de subportadoras
                      adyacentes (k, k+1) lleva los símbolos (s0, s1) y, por la 2ª antena,
                      (-s1*, s0*). Cada antena se escala por 1/√2 para REPARTIR la potencia total
                      entre las dos (potencia total constante -> penalización de ~3 dB vs MRC).
                      Cada antena RX ve dos canales independientes (H1, H2) y se decodifica con el
                      combinador ortogonal de Alamouti, que separa s0 y s1 sin interferencia:
                        s0_est = Σ_rx (conj(h1)·r0 + h2·conj(r1)) / Σ_rx (|h1|²+|h2|²)
                        s1_est = Σ_rx (conj(h1)·r1 - h2·conj(r0)) / Σ_rx (|h1|²+|h2|²)
                      Diversidad = 2·n_rx (2x1 -> 2, 2x2 -> 4).

    A diferencia de OFDM, SFBC NO intercala pilotos: usa TODAS las subportadoras en pares (el CSI
    es genie en este simulador, igual criterio que SC-FDM). Así los dos elementos de cada par son
    físicamente adyacentes y se cumple H[k] ≈ H[k+1] (el canal Pedestrian A es muy suave en
    frecuencia). El único par que cruza el hueco de DC tiene un desajuste de canal despreciable.
    """
    mod = MODULACIONES[modulacion]
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    fs = params["fs"]
    bps = mod["bits"]

    n_par = n_sc // 2                                  # Nº de pares Alamouti por símbolo OFDM
    n_datos = n_par * 2                                # Subportadoras de datos usadas (siempre par)

    # --- TX: bits -> símbolos QAM (relleno a múltiplo de bps y de n_datos), igual que cadena_tx_rx ---
    falta = (-len(bits_tx)) % bps                      # Bits que faltan para múltiplo de bps
    if falta:
        bits_tx_pad = np.concatenate([bits_tx, np.zeros(falta, dtype=np.uint8)])
    else:
        bits_tx_pad = bits_tx
    simbolos_qam = mod["mapear"](bits_tx_pad)          # Mapea todos los bits a símbolos QAM
    falta_sim = (-len(simbolos_qam)) % n_datos         # Completa un nº entero de símbolos OFDM
    if falta_sim:
        simbolos_qam = np.concatenate([simbolos_qam, np.zeros(falta_sim, dtype=complex)])
    n_ofdm = len(simbolos_qam) // n_datos              # Nº de símbolos OFDM a transmitir

    constelacion_tx = simbolos_qam.copy() if capturar_constelaciones else None
    simbolos_rx_datos = []                             # Acumulador de símbolos RX (constelación)
    bits_rx_total = []                                 # Acumulador de bits recibidos

    # Índices FFT de las n_sc subportadoras activas (no dependen de los datos: rejilla vacía)
    _, indices_fft = modulacion_ofdm(np.zeros(n_sc, dtype=complex), n_fft, n_cp)
    esc = 1.0 / np.sqrt(2.0)                           # Reparto de potencia entre las 2 antenas TX

    for i in range(n_ofdm):                            # Procesa símbolo a símbolo
        bloque = simbolos_qam[i * n_datos:(i + 1) * n_datos]  # Bloque de n_datos símbolos
        s0 = bloque[0::2]                              # Símbolo "par" de cada pareja (subportadora k)
        s1 = bloque[1::2]                              # Símbolo "impar" de cada pareja (subportadora k+1)

        # Potencia de TX de referencia (1 antena a plena potencia con estos datos) para el piso de
        # ruido fijo de la Práctica 5; None -> calibración clásica al recibido (comportamiento P4).
        if ruido_piso_fijo:
            grid_ref = np.zeros(n_sc, dtype=complex)
            grid_ref[0::2][:n_par] = s0
            grid_ref[1::2][:n_par] = s1
            senal_ref, _ = modulacion_ofdm(grid_ref, n_fft, n_cp)
            pot_ref = float(np.mean(np.abs(senal_ref) ** 2))
        else:
            pot_ref = None

        if n_tx <= 1:
            # --- SISO de referencia (1 antena TX): símbolos en todas las subportadoras + ZF/MRC ---
            x = np.zeros(n_sc, dtype=complex)
            x[0::2][:n_par] = s0
            x[1::2][:n_par] = s1
            num = np.zeros(n_sc, dtype=complex)        # Σ conj(H)·Y  (ZF si n_rx=1, MRC si n_rx>1)
            den = np.zeros(n_sc, dtype=float)          # Σ |H|²
            for _r in range(max(n_rx, 1)):
                H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)
                senal_canal, _ = modulacion_ofdm(x * H, n_fft, n_cp)
                senal_rx = _agregar_ruido(senal_canal, snr_db, rng, pot_ref)
                rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
                num += np.conj(H) * rejilla_rx
                den += np.abs(H) ** 2
            rejilla_eq = num / (den + 1e-12)           # Estimación del símbolo (ZF/MRC)
            datos_rx = np.empty(n_datos, dtype=complex)
            datos_rx[0::2] = rejilla_eq[0::2][:n_par]
            datos_rx[1::2] = rejilla_eq[1::2][:n_par]
        else:
            # --- SFBC Alamouti (2 antenas TX): rejillas espacio-frecuencia x1 y x2 ---
            x1 = np.zeros(n_sc, dtype=complex)         # Rejilla de la antena 1
            x2 = np.zeros(n_sc, dtype=complex)         # Rejilla de la antena 2
            x1[0::2][:n_par] = esc * s0                # Antena 1, subportadora k   ->  s0
            x1[1::2][:n_par] = esc * s1                # Antena 1, subportadora k+1 ->  s1
            x2[0::2][:n_par] = esc * (-np.conj(s1))    # Antena 2, subportadora k   -> -s1*
            x2[1::2][:n_par] = esc * (np.conj(s0))     # Antena 2, subportadora k+1 ->  s0*

            num0 = np.zeros(n_par, dtype=complex)      # Acumulador del estimador de s0
            num1 = np.zeros(n_par, dtype=complex)      # Acumulador del estimador de s1
            den = np.zeros(n_par, dtype=float)         # Σ (|h1|²+|h2|²) sobre las antenas RX
            for _r in range(n_rx):
                H1 = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)  # Canal TX1 -> esta RX
                H2 = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)  # Canal TX2 -> esta RX
                senal1, _ = modulacion_ofdm(x1 * H1, n_fft, n_cp)            # Aporte de la antena 1
                senal2, _ = modulacion_ofdm(x2 * H2, n_fft, n_cp)            # Aporte de la antena 2
                senal_rx = _agregar_ruido(senal1 + senal2, snr_db, rng, pot_ref)  # Se suman + AWGN
                rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
                r0 = rejilla_rx[0::2][:n_par]          # Recibido en la subportadora k
                r1 = rejilla_rx[1::2][:n_par]          # Recibido en la subportadora k+1
                h1 = H1[0::2][:n_par]                  # Canal TX1 en el par (H1[k] ≈ H1[k+1])
                h2 = H2[0::2][:n_par]                  # Canal TX2 en el par
                num0 += np.conj(h1) * r0 + h2 * np.conj(r1)   # Combinador Alamouti para s0
                num1 += np.conj(h1) * r1 - h2 * np.conj(r0)   # Combinador Alamouti para s1
                den += np.abs(h1) ** 2 + np.abs(h2) ** 2
            # /esc deshace el reparto de potencia 1/√2 y devuelve la constelación a escala unitaria
            # (imprescindible para que el demapeador de 16/64-QAM use las regiones de decisión correctas).
            s0_est = (num0 / (den + 1e-12)) / esc
            s1_est = (num1 / (den + 1e-12)) / esc
            datos_rx = np.empty(n_datos, dtype=complex)
            datos_rx[0::2] = s0_est
            datos_rx[1::2] = s1_est

        if capturar_constelaciones:
            simbolos_rx_datos.append(datos_rx)
        bits_rx_total.append(mod["demapear"](datos_rx))   # Demapea a bits y acumula

    bits_rx = np.concatenate(bits_rx_total) if bits_rx_total else np.array([], dtype=np.uint8)
    bits_rx = bits_rx[:len(bits_tx_pad)]               # Recorta al tamaño transmitido
    ber = calcular_ber(bits_tx_pad, bits_rx)           # BER

    salida = {
        "bits_rx": bits_rx[:len(bits_tx)],             # Bits útiles (sin padding inicial)
        "ber": ber,
        "n_simbolos_ofdm": n_ofdm,
    }
    if capturar_constelaciones:
        salida["constelacion_tx"] = constelacion_tx
        salida["constelacion_rx"] = (np.concatenate(simbolos_rx_datos)
                                     if simbolos_rx_datos else np.array([], dtype=complex))
    return salida


# =====================================================================================
# === [BLOQUE 6b] BEAMFORMING — Práctica 5 (precodificación MRT con CSI en el TX)    ===
# =====================================================================================
# Beamforming en TX de "baja correlación" (diapositivas Cap. 5, pág. 170-179): conociendo el
# canal en el transmisor (CSI en TX), cada subportadora se multiplica por un VECTOR DE
# PRECODIFICACIÓN
#       w̄[k] = h̄[k]* / ‖h̄[k]‖            (MRT, Maximum Ratio Transmission)
# Es el DUAL EXACTO de MRC (Práctica 3): MRC combina en RX con conj(H); MRT precodifica en TX
# con conj(H) normalizado. El vector rota la fase de cada antena para que las señales lleguen
# ALINEADAS EN FASE al RX y asigna más potencia a las antenas con mejor |h_i|, manteniendo la
# potencia total de TX (‖w̄‖=1). Resultado: GANANCIA DE ARREGLO (la potencia RX crece ×N_T) +
# DIVERSIDAD de orden N_T. Como SFBC, NO usa pilotos: el CSI es genie (imprescindible, el TX
# necesita el canal para precodificar). En OFDM la precodificación es POR SUBPORTADORA (cada
# subportadora ve un canal plano), tal como indica la pág. 178-179.

def precodificar_mrt(s_grid: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    PASO CLAVE EN EL TX. Entran los símbolos s_grid (n_sc) y la matriz de canal H (n_tx × n_sc,
    una fila por antena TX) y salen la rejilla transmitida por cada antena X (n_tx × n_sc) y la
    ganancia efectiva del enlace 'norma' (n_sc):
        norma[k] = ‖h̄[k]‖ = sqrt(Σ_i |H_i[k]|²)
        w_i[k]   = conj(H_i[k]) / norma[k]      (cada columna tiene norma 1 -> potencia constante)
        X_i[k]   = w_i[k] · s_grid[k]
    Tras el canal, el RX recibe Σ_i H_i[k]·X_i[k] = norma[k]·s_grid[k] (las antenas suman en fase).
    """
    norma = np.sqrt(np.sum(np.abs(H) ** 2, axis=0))       # ‖h̄[k]‖ por subportadora
    W = np.conj(H) / (norma + 1e-12)                       # Pesos MRT (cada columna de norma 1)
    X = W * s_grid[np.newaxis, :]                          # Rejilla por antena: X_i[k] = w_i[k]·s[k]
    return X, norma


def detector_beamforming(R: np.ndarray, norma: np.ndarray) -> np.ndarray:
    """
    PASO CLAVE EN EL RX. Entra la rejilla recibida R (n_sc) y la ganancia efectiva 'norma'
    (n_sc) y sale la estimación del símbolo: ŝ[k] = R[k] / norma[k]. Como el precodificador ya
    alineó las fases, basta normalizar por ‖h̄[k]‖ (no hay fase residual que deshacer).
    """
    return R / (norma + 1e-12)


def cadena_tx_beamforming_rx(bits_tx: np.ndarray, modulacion: str, params: Dict,
                              snr_db: float, rng: np.random.Generator,
                              capturar_constelaciones: bool = False,
                              n_tx: int = 2) -> Dict:
    """
    Núcleo de la Práctica 5 (beamforming en TX con MRT). El TX conoce el canal y precodifica
    hacia 1 antena RX con w̄[k]=h̄[k]*/‖h̄[k]‖ (la ganancia/diversidad de RX la cubre la P3-MRC):

      - n_tx == 1  -> SISO de referencia (sin beamforming): canal Pedestrian A + ecualización ZF.
      - n_tx >= 2  -> beamforming MRT: ganancia de arreglo (×n_tx en potencia RX) + diversidad
                      de orden n_tx. La curva BER vs SNR baja con MAYOR PENDIENTE (diversidad) y
                      se corre a la IZQUIERDA ~10·log10(n_tx) dB (ganancia de arreglo).

    Igual que SFBC, NO intercala pilotos: TODAS las subportadoras llevan datos y el CSI es genie.
    El ruido usa un PISO FIJO referido a la potencia de TX de una antena (ver _agregar_ruido),
    para que la ganancia de arreglo sea visible y la comparación con SFBC/MRC sea justa.
    """
    mod = MODULACIONES[modulacion]
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    fs = params["fs"]
    bps = mod["bits"]

    # --- TX: bits -> símbolos QAM (relleno a múltiplo de bps y de n_sc), igual que cadena_tx_sfbc_rx ---
    falta = (-len(bits_tx)) % bps                       # Bits que faltan para múltiplo de bps
    if falta:
        bits_tx_pad = np.concatenate([bits_tx, np.zeros(falta, dtype=np.uint8)])
    else:
        bits_tx_pad = bits_tx
    simbolos_qam = mod["mapear"](bits_tx_pad)           # Mapea todos los bits a símbolos QAM
    falta_sim = (-len(simbolos_qam)) % n_sc             # Completa un nº entero de símbolos OFDM
    if falta_sim:
        simbolos_qam = np.concatenate([simbolos_qam, np.zeros(falta_sim, dtype=complex)])
    n_ofdm = len(simbolos_qam) // n_sc                  # Nº de símbolos OFDM a transmitir

    constelacion_tx = simbolos_qam.copy() if capturar_constelaciones else None
    simbolos_rx_datos = []                              # Acumulador de símbolos RX (constelación)
    bits_rx_total = []                                  # Acumulador de bits recibidos

    # Índices FFT de las n_sc subportadoras activas (no dependen de los datos: rejilla vacía)
    _, indices_fft = modulacion_ofdm(np.zeros(n_sc, dtype=complex), n_fft, n_cp)

    for i in range(n_ofdm):                             # Procesa símbolo a símbolo
        s_grid = simbolos_qam[i * n_sc:(i + 1) * n_sc]  # n_sc símbolos en TODAS las subportadoras

        # Canal: un enlace Pedestrian A INDEPENDIENTE por antena TX (caso de baja correlación)
        H = np.stack([generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)
                      for _ in range(max(n_tx, 1))])    # (n_tx × n_sc)

        # Potencia de transmisión de referencia: la de una sola antena enviando s_grid (SISO)
        senal_ref, _ = modulacion_ofdm(s_grid, n_fft, n_cp)
        pot_ref = float(np.mean(np.abs(senal_ref) ** 2))

        if n_tx <= 1:
            # --- SISO de referencia (1 antena TX): canal + ecualización ZF ---
            H0 = H[0]
            senal_canal, _ = modulacion_ofdm(s_grid * H0, n_fft, n_cp)
            senal_rx = _agregar_ruido(senal_canal, snr_db, rng, pot_ref)
            rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
            datos_rx = rejilla_rx / (H0 + 1e-12)        # Ecualización ZF
        else:
            # --- Beamforming MRT (n_tx antenas TX, 1 RX) ---
            X, norma = precodificar_mrt(s_grid, H)      # ← PASO CLAVE TX: rejillas precodificadas
            senal_rx_clean = sum(modulacion_ofdm(X[a] * H[a], n_fft, n_cp)[0]
                                 for a in range(n_tx))  # El RX ve la suma coherente de las n_tx antenas
            senal_rx = _agregar_ruido(senal_rx_clean, snr_db, rng, pot_ref)
            rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
            datos_rx = detector_beamforming(rejilla_rx, norma)  # ← PASO CLAVE RX: ŝ = R/‖h̄‖

        if capturar_constelaciones:
            simbolos_rx_datos.append(datos_rx)
        bits_rx_total.append(mod["demapear"](datos_rx))  # Demapea a bits y acumula

    bits_rx = np.concatenate(bits_rx_total) if bits_rx_total else np.array([], dtype=np.uint8)
    bits_rx = bits_rx[:len(bits_tx_pad)]                # Recorta al tamaño transmitido
    ber = calcular_ber(bits_tx_pad, bits_rx)            # BER

    salida = {
        "bits_rx": bits_rx[:len(bits_tx)],              # Bits útiles (sin padding inicial)
        "ber": ber,
        "n_simbolos_ofdm": n_ofdm,
    }
    if capturar_constelaciones:
        salida["constelacion_tx"] = constelacion_tx
        salida["constelacion_rx"] = (np.concatenate(simbolos_rx_datos)
                                     if simbolos_rx_datos else np.array([], dtype=complex))
    return salida


# =====================================================================================
# === [BLOQUE 6c] MIMO — DIVERSIDAD COMBINADA TX + RX — Práctica 6 (SFBC + MRC)      ===
# =====================================================================================
# Esta práctica MEZCLA las dos técnicas de diversidad ya vistas para lograr la MÁXIMA
# diversidad de un sistema MIMO, enviando la MISMA información (redundante) por TODAS las
# antenas (NO flujos independientes):
#   · DIVERSIDAD EN TX  -> SFBC/Alamouti (P4): 2 antenas TX envían, por cada par (k,k+1),
#     (s0,s1) y su versión ortogonal (-s1*, s0*), cada una escalada por 1/√2.
#   · DIVERSIDAD EN RX  -> MRC (P3): las N_R antenas RX se combinan sumando coherentemente,
#     ponderando por la ganancia de canal.
# El receptor hace ambas a la vez: por cada antena RX aplica el combinador ortogonal de
# Alamouti (recupera la diversidad TX) y SUMA las N_R antenas ponderando por Σ(|h1|²+|h2|²)
# (eso es MRC -> recupera la diversidad RX). Orden de diversidad total = N_T·N_R = 2·N_R:
#   2x1 -> orden 2,  2x2 -> orden 4,  2x4 -> orden 8. La curva BER cae con mayor pendiente
# al subir la orden. La versión didáctica y comentada paso a paso está en mimo.py.

def cadena_tx_mimo_div_rx(bits_tx: np.ndarray, modulacion: str, params: Dict,
                          snr_db: float, rng: np.random.Generator,
                          capturar_constelaciones: bool = False,
                          n_tx: int = 2, n_rx: int = 2,
                          ruido_piso_fijo: bool = False) -> Dict:
    """
    Núcleo de la Práctica 6 (diversidad combinada TX+RX). Combina SFBC (Alamouti, 2 antenas
    TX) con MRC (N_R antenas RX) enviando la MISMA información redundante por todas las antenas:

      - n_tx == 1  -> SISO/MRC de referencia (solo diversidad RX): canal Pedestrian A y
                      combinación MRC sobre n_rx antenas (ZF si n_rx=1). Orden = n_rx.
      - n_tx == 2  -> SFBC + MRC: Alamouti sobre 2 antenas TX (cada una a 1/√2 de potencia) y
                      combinación conjunta Alamouti+MRC sobre las n_rx antenas RX. La orden de
                      diversidad es 2·n_rx (2x1 -> 2, 2x2 -> 4, 2x4 -> 8): la curva BER cae con
                      MAYOR PENDIENTE al subir n_rx.

    Como SFBC/BF, NO intercala pilotos: usa TODAS las subportadoras en pares (CSI genie), con
    H[k] ≈ H[k+1] al ser adyacentes. El ruido se calibra al recibido (pot_ref=None) salvo que
    se pida piso fijo, para comparar la ORDEN de diversidad a igual SNR recibida.
    """
    mod = MODULACIONES[modulacion]
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    fs = params["fs"]
    bps = mod["bits"]

    n_par = n_sc // 2                                   # Nº de pares Alamouti por símbolo OFDM
    n_datos = n_par * 2                                 # Subportadoras de datos usadas (siempre par)

    # --- TX: bits -> símbolos QAM (relleno a múltiplo de bps y de n_datos) ---
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
    simbolos_rx_datos = []
    bits_rx_total = []

    _, indices_fft = modulacion_ofdm(np.zeros(n_sc, dtype=complex), n_fft, n_cp)
    esc = 1.0 / np.sqrt(2.0)                            # Reparto de potencia entre las 2 antenas TX

    for i in range(n_ofdm):
        bloque = simbolos_qam[i * n_datos:(i + 1) * n_datos]
        s0 = bloque[0::2]                              # Símbolo "par" del par (subportadora k)
        s1 = bloque[1::2]                              # Símbolo "impar" del par (subportadora k+1)

        # Potencia de TX de referencia (1 antena) para el piso de ruido fijo opcional.
        if ruido_piso_fijo:
            grid_ref = np.zeros(n_sc, dtype=complex)
            grid_ref[0::2][:n_par] = s0
            grid_ref[1::2][:n_par] = s1
            senal_ref, _ = modulacion_ofdm(grid_ref, n_fft, n_cp)
            pot_ref = float(np.mean(np.abs(senal_ref) ** 2))
        else:
            pot_ref = None

        if n_tx <= 1:
            # --- Referencia: solo diversidad RX (MRC sobre n_rx antenas; ZF si n_rx=1) ---
            x = np.zeros(n_sc, dtype=complex)
            x[0::2][:n_par] = s0
            x[1::2][:n_par] = s1
            num = np.zeros(n_sc, dtype=complex)        # Σ conj(H)·Y  (MRC en RX)
            den = np.zeros(n_sc, dtype=float)          # Σ |H|²
            for _r in range(max(n_rx, 1)):
                H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)
                senal_canal, _ = modulacion_ofdm(x * H, n_fft, n_cp)
                senal_rx = _agregar_ruido(senal_canal, snr_db, rng, pot_ref)
                rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
                num += np.conj(H) * rejilla_rx
                den += np.abs(H) ** 2
            rejilla_eq = num / (den + 1e-12)
            datos_rx = np.empty(n_datos, dtype=complex)
            datos_rx[0::2] = rejilla_eq[0::2][:n_par]
            datos_rx[1::2] = rejilla_eq[1::2][:n_par]
        else:
            # --- SFBC (Alamouti, diversidad TX) + MRC (diversidad RX): la MEZCLA ---
            x1 = np.zeros(n_sc, dtype=complex)         # Rejilla de la antena TX 1
            x2 = np.zeros(n_sc, dtype=complex)         # Rejilla de la antena TX 2
            x1[0::2][:n_par] = esc * s0                # Antena 1, subportadora k   ->  s0
            x1[1::2][:n_par] = esc * s1                # Antena 1, subportadora k+1 ->  s1
            x2[0::2][:n_par] = esc * (-np.conj(s1))    # Antena 2, subportadora k   -> -s1*
            x2[1::2][:n_par] = esc * (np.conj(s0))     # Antena 2, subportadora k+1 ->  s0*

            num0 = np.zeros(n_par, dtype=complex)      # Acumulador del estimador de s0
            num1 = np.zeros(n_par, dtype=complex)      # Acumulador del estimador de s1
            den = np.zeros(n_par, dtype=float)         # Σ (|h1|²+|h2|²) sobre las antenas RX (MRC)
            for _r in range(n_rx):                     # ← SUMA sobre antenas RX = MRC (diversidad RX)
                H1 = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)  # Canal TX1 -> esta RX
                H2 = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)  # Canal TX2 -> esta RX
                senal1, _ = modulacion_ofdm(x1 * H1, n_fft, n_cp)            # Aporte de la antena 1
                senal2, _ = modulacion_ofdm(x2 * H2, n_fft, n_cp)            # Aporte de la antena 2
                senal_rx = _agregar_ruido(senal1 + senal2, snr_db, rng, pot_ref)  # Se suman + AWGN
                rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
                r0 = rejilla_rx[0::2][:n_par]          # Recibido en la subportadora k
                r1 = rejilla_rx[1::2][:n_par]          # Recibido en la subportadora k+1
                h1 = H1[0::2][:n_par]                  # Canal TX1 en el par (H1[k] ≈ H1[k+1])
                h2 = H2[0::2][:n_par]                  # Canal TX2 en el par
                num0 += np.conj(h1) * r0 + h2 * np.conj(r1)   # ← combinación Alamouti (diversidad TX)
                num1 += np.conj(h1) * r1 - h2 * np.conj(r0)
                den += np.abs(h1) ** 2 + np.abs(h2) ** 2
            # /esc deshace el reparto de potencia 1/√2 (devuelve la escala unitaria para el demapeo)
            s0_est = (num0 / (den + 1e-12)) / esc
            s1_est = (num1 / (den + 1e-12)) / esc
            datos_rx = np.empty(n_datos, dtype=complex)
            datos_rx[0::2] = s0_est
            datos_rx[1::2] = s1_est

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
        salida["constelacion_rx"] = (np.concatenate(simbolos_rx_datos)
                                     if simbolos_rx_datos else np.array([], dtype=complex))
    return salida


# =====================================================================================
# === [BLOQUE 6d] CODIFICACIÓN DE CANAL — Práctica 7 (convolucional + turbo)         ===
# =====================================================================================
# Implementa los DOS códigos correctores de errores (FEC) de las diapositivas
# "Codificación de Canal" (Cap. 5 parte 2, diapos 4-37), tal como los especifica LTE:
#   · Convolucional tail-biting tasa 1/3: G0=133, G1=171, G2=165 (octal), 6 memorias;
#     se decodifica con Viterbi suave de envoltura (tail-biting).
#   · Turbo tasa 1/3: dos RSC de 8 estados (g0=1+D²+D³, g1=1+D+D³) + entrelazador QPP;
#     se decodifica con dos decodificadores APP (BCJR) iterativos (decisión suave).
# La GANANCIA DE CODIFICACIÓN se logra con DECISIÓN SUAVE: tras la ecualización ZF se
# calcula un LLR por bit ponderado por la calidad |H[k]|² de su subportadora. Los
# decodificadores trabajan POR LOTES (B bloques a la vez) para que el sitio sea ágil.
# La versión didáctica y comentada paso a paso está en codificacion.py.

_G_CONV = (0o133, 0o171, 0o165)                                     # generadores convolucionales
# Entrelazador QPP: π(i)=(f1·i+f2·i²) mod K (k=40…6144 en LTE; aquí un subconjunto).
_QPP_TABLA = {40: (3, 10), 512: (31, 64), 1024: (31, 84), 2048: (31, 64), 6144: (263, 480)}
K_TURBO = 512                                                       # tamaño de bloque
PAYLOAD_BLOQUE = K_TURBO                                            # bits de info por bloque (sin CRC)


# ---- Trellis y predecesores (se calculan una sola vez) ---------------------------------
def _preds_cc(nxt, n):
    ps = np.zeros((n, 2), np.int64); pu = np.zeros((n, 2), np.int64); cnt = np.zeros(n, np.int64)
    for s in range(n):
        for u in range(2):
            ns = nxt[s, u]; ps[ns, cnt[ns]] = s; pu[ns, cnt[ns]] = u; cnt[ns] += 1
    return ps, pu


def _conv_trellis_cc():
    taps = np.array([[(g >> (6 - j)) & 1 for j in range(7)] for g in _G_CONV], dtype=np.int64)
    nxt = np.zeros((64, 2), np.int64); out = np.zeros((64, 2, 3), np.int64)
    for s in range(64):
        Ds = [(s >> i) & 1 for i in range(6)]
        for u in range(2):
            for gi in range(3):
                o = taps[gi, 0] & u
                for i in range(6):
                    o ^= taps[gi, i + 1] & Ds[i]
                out[s, u, gi] = o & 1
            nxt[s, u] = ((s << 1) | u) & 0x3F
    return nxt, out


def _rsc_trellis_cc():
    nxt = np.zeros((8, 2), np.int64); par = np.zeros((8, 2), np.int64)
    for s in range(8):
        m1, m2, m3 = s & 1, (s >> 1) & 1, (s >> 2) & 1
        for u in range(2):
            a = u ^ m2 ^ m3                                         # realimentación 1+D²+D³
            par[s, u] = a ^ m1 ^ m3                                 # paridad 1+D+D³
            nxt[s, u] = a | (m1 << 1) | (m2 << 2)
    return nxt, par


_NXT_CC, _OUT_CC = _conv_trellis_cc()
_SIGN_CC = (1 - 2 * _OUT_CC).astype(float)                          # (64,2,3) salida en ±1
_PS_CC, _PU_CC = _preds_cc(_NXT_CC, 64)
_NXT_RC, _PAR_RC = _rsc_trellis_cc()
_SP_RC = (1 - 2 * _PAR_RC).astype(float)                           # (8,2) paridad en ±1
_PS_RC, _PU_RC = _preds_cc(_NXT_RC, 8)
_SU_RC = np.array([1.0, -1.0])
_NEG = -1e9


def qpp_interleaver(K):
    f1, f2 = _QPP_TABLA[K]
    i = np.arange(K)
    return (f1 * i + f2 * i * i) % K


# ---- Codificadores por lotes -----------------------------------------------------------
def codificar_convolucional_lote(M):
    """M (B×K) -> bits de código (B×3K), tail-biting (estado inicial = últimos 6 bits)."""
    B, K = M.shape
    estado = np.zeros(B, np.int64)
    for i in range(6):
        estado |= (M[:, K - 1 - i].astype(np.int64) << i)
    out = np.empty((B, K, 3), np.uint8)
    for k in range(K):
        u = M[:, k].astype(np.int64)
        out[:, k, :] = _OUT_CC[estado, u]
        estado = _NXT_CC[estado, u]
    return out.reshape(B, 3 * K)


def _rsc_encode_lote(M):
    B, K = M.shape
    s = np.zeros(B, np.int64)
    paridad = np.empty((B, K), np.uint8)
    for k in range(K):
        u = M[:, k].astype(np.int64)
        paridad[:, k] = _PAR_RC[s, u]
        s = _NXT_RC[s, u]
    cs = np.empty((B, 3), np.uint8); cp = np.empty((B, 3), np.uint8)
    for j in range(3):
        u = ((s >> 1) & 1) ^ ((s >> 2) & 1)                        # entrada que termina el trellis
        cs[:, j] = u; cp[:, j] = _PAR_RC[s, u]; s = _NXT_RC[s, u]
    return paridad, cs, cp


def codificar_turbo_lote(M):
    """M (B×K) -> (vector serializado (B×(3K+12)), permutación QPP). Tasa ≈ 1/3."""
    B, K = M.shape
    perm = qpp_interleaver(K)
    z1, t1s, t1p = _rsc_encode_lote(M)
    z2, t2s, t2p = _rsc_encode_lote(M[:, perm])
    vec = np.concatenate([M, z1, z2, t1s, t1p, t2s, t2p], axis=1)
    return vec, perm


# ---- Demapeo suave -> LLR (ponderado por |H|²) -----------------------------------------
def llr_qpsk(s_eq, g):
    s = s_eq * np.sqrt(2)
    L = np.empty(2 * len(s_eq))
    L[0::2] = g * np.real(s)
    L[1::2] = g * np.imag(s)
    return L


def _llr_eje_4pam(x, g):
    niveles = np.array([-3, -1, 1, 3])
    b0 = np.array([0, 0, 1, 1]); b1 = np.array([0, 1, 1, 0])
    d2 = (x[:, None] - niveles[None, :]) ** 2
    L0 = d2[:, b0 == 1].min(axis=1) - d2[:, b0 == 0].min(axis=1)
    L1 = d2[:, b1 == 1].min(axis=1) - d2[:, b1 == 0].min(axis=1)
    return g * L0, g * L1


def llr_16qam(s_eq, g):
    s = s_eq * np.sqrt(10)
    L0i, L1i = _llr_eje_4pam(np.real(s), g)
    L0q, L1q = _llr_eje_4pam(np.imag(s), g)
    L = np.empty(4 * len(s_eq))
    L[0::4], L[1::4] = L0i, L1i
    L[2::4], L[3::4] = L0q, L1q
    return L


# ---- Decodificadores por lotes ---------------------------------------------------------
def viterbi_lote(LLR, vueltas=4):
    """LLR (B×3K) -> bits (B×K). Viterbi suave tail-biting (envoltura de varias vueltas)."""
    B = LLR.shape[0]; K = LLR.shape[1] // 3
    LLR = LLR.reshape(B, K, 3)
    LLRv = np.concatenate([LLR] * vueltas, axis=1); T = vueltas * K
    metrica = np.zeros((B, 64))
    psd = np.empty((T, B, 64), np.int16); psu = np.empty((T, B, 64), np.int8)
    p0s, p1s = _PS_CC[:, 0], _PS_CC[:, 1]; p0u, p1u = _PU_CC[:, 0], _PU_CC[:, 1]
    for t in range(T):
        bm = np.tensordot(LLRv[:, t, :], _SIGN_CC, axes=([1], [2]))     # (B,64,2)
        c0 = metrica[:, p0s] + bm[:, p0s, p0u]
        c1 = metrica[:, p1s] + bm[:, p1s, p1u]
        ch = c1 > c0
        metrica = np.where(ch, c1, c0)
        psd[t] = np.where(ch, p1s[None, :], p0s[None, :])
        psu[t] = np.where(ch, p1u[None, :], p0u[None, :])
    s = np.argmax(metrica, axis=1); bits = np.empty((T, B), np.uint8); ar = np.arange(B)
    for t in range(T - 1, -1, -1):
        bits[t] = psu[t, ar, s]; s = psd[t, ar, s]
    return bits[(vueltas - 1) * K:].T


def _bcjr_lote(Lsys, Lpar, La, term_sys, term_par):
    B, K = Lsys.shape
    Ls = np.concatenate([Lsys, term_sys], axis=1)
    Lp = np.concatenate([Lpar, term_par], axis=1)
    Lap = np.concatenate([La, np.zeros((B, 3))], axis=1)
    T = K + 3
    Aa = 0.5 * (Lap + Ls)
    gamma = (Aa.T[:, :, None, None] * _SU_RC[None, None, None, :]
             + 0.5 * _SP_RC[None, None, :, :] * Lp.T[:, :, None, None])    # (T,B,8,2)
    alpha = np.full((T + 1, B, 8), _NEG); alpha[0, :, 0] = 0.0
    for k in range(T):
        alpha[k + 1] = (alpha[k][:, _PS_RC] + gamma[k][:, _PS_RC, _PU_RC]).max(axis=2)
    beta = np.full((T + 1, B, 8), _NEG); beta[T, :, 0] = 0.0
    for k in range(T - 1, -1, -1):
        beta[k] = (beta[k + 1][:, _NXT_RC] + gamma[k]).max(axis=2)
    full = alpha[:K][:, :, :, None] + gamma[:K] + beta[1:K + 1][:, :, _NXT_RC]    # (K,B,8,2)
    m0 = full[:, :, :, 0].max(axis=2); m1 = full[:, :, :, 1].max(axis=2)
    return (m0 - m1).T - La - Lsys


def turbo_lote(Lsys, Lp1, Lp2, Lt1s, Lt1p, Lt2s, Lt2p, perm, K, n_iter=6):
    """Decodificación turbo iterativa por lotes -> bits (B×K)."""
    B = Lsys.shape[0]
    invperm = np.empty(K, np.int64); invperm[perm] = np.arange(K)
    La1 = np.zeros((B, K)); Le1 = np.zeros((B, K))
    for _ in range(n_iter):
        Le1 = _bcjr_lote(Lsys, Lp1, La1, Lt1s, Lt1p)
        Le2 = _bcjr_lote(Lsys[:, perm], Lp2, Le1[:, perm], Lt2s, Lt2p)
        La1 = Le2[:, invperm]
    return (Lsys + Le1 + La1 < 0).astype(np.uint8)


# ---- Cadena completa de la Práctica 7 --------------------------------------------------
def _ofdm_canal_zf(simbolos, params, snr_db, rng):
    """TX OFDM (todas las subportadoras, sin pilotos) -> canal Pedestrian A + AWGN -> RX +
    ZF. Devuelve los símbolos ecualizados ŝ, la ganancia |H|² por símbolo y nº de símbolos
    OFDM. Cada símbolo OFDM ve un canal independiente (diversidad en frecuencia)."""
    n_sc, n_fft, n_cp, fs = params["n_sc"], params["n_fft"], params["n_cp"], params["fs"]
    falta = (-len(simbolos)) % n_sc
    if falta:
        simbolos = np.concatenate([simbolos, np.zeros(falta, dtype=complex)])
    n_ofdm = len(simbolos) // n_sc
    s_eq, g = [], []
    for i in range(n_ofdm):
        rejilla = simbolos[i * n_sc:(i + 1) * n_sc]
        _, indices_fft = mapeo_sc_a_fft(rejilla, n_fft)
        H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)
        senal_canal, _ = modulacion_ofdm(rejilla * H, n_fft, n_cp)
        senal_rx = agregar_ruido_awgn(senal_canal, snr_db, rng)
        rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
        s_eq.append(rejilla_rx / H)
        g.append(np.abs(H) ** 2)
    return np.concatenate(s_eq), np.concatenate(g), n_ofdm


def cadena_tx_codif_rx(bits_tx, modulacion, codigo, params, snr_db, rng,
                       capturar_constelaciones=False):
    """
    Núcleo de la Práctica 7 (codificación de canal). Según `codigo`:
      - "ninguno"       -> referencia SIN codificar (mapeo directo + decisión dura).
      - "convolucional" -> convolucional tail-biting tasa 1/3 + Viterbi suave.
      - "turbo"         -> turbo tasa 1/3 (QPP) + APP/BCJR iterativo.
    Todos sobre la misma cadena OFDM + Pedestrian A. La info se segmenta en bloques de
    K_TURBO bits. Devuelve los bits recuperados, la BER y (en los códigos) el nº de bloques.
    """
    mapear = qpsk_mapear if modulacion == "QPSK" else qam16_mapear
    demapear = qpsk_demapear if modulacion == "QPSK" else qam16_demapear
    llr_fn = llr_qpsk if modulacion == "QPSK" else llr_16qam
    bps = 2 if modulacion == "QPSK" else 4

    # ---------- Referencia SIN codificar ----------
    if codigo == "ninguno":
        b = bits_tx
        falta = (-len(b)) % bps
        if falta:
            b = np.concatenate([b, np.zeros(falta, dtype=np.uint8)])
        s_eq, g, n_ofdm = _ofdm_canal_zf(mapear(b), params, snr_db, rng)
        bits_rx = demapear(s_eq)[:len(bits_tx)]
        salida = {"bits_rx": bits_rx, "ber": calcular_ber(bits_tx, bits_rx),
                  "n_simbolos_ofdm": n_ofdm, "n_bloques": 0}
        if capturar_constelaciones:
            salida["constelacion_rx"] = s_eq[:min(len(s_eq), 4000)]
        return salida

    # ---------- TRANSMISIÓN: segmentación en bloques de K + codificación ----------
    K = K_TURBO
    n_bloques = max(1, int(np.ceil(len(bits_tx) / K)))
    payload = np.concatenate([bits_tx, np.zeros(n_bloques * K - len(bits_tx), dtype=np.uint8)])
    bloques = payload.reshape(n_bloques, K)

    if codigo == "convolucional":
        cod = codificar_convolucional_lote(bloques)                # (B, 3K)
        perm = None
    else:
        cod, perm = codificar_turbo_lote(bloques)                  # (B, 3K+12)
    code_len = cod.shape[1]
    bits_cod = cod.reshape(-1)                                      # serie de bits de código

    # ---------- QAM + OFDM + canal + ZF + demapeo suave ----------
    falta = (-len(bits_cod)) % bps
    if falta:
        bits_cod = np.concatenate([bits_cod, np.zeros(falta, dtype=np.uint8)])
    s_eq, g, n_ofdm = _ofdm_canal_zf(mapear(bits_cod), params, snr_db, rng)
    llr = llr_fn(s_eq, g)[:n_bloques * code_len].reshape(n_bloques, code_len)

    # ---------- Decodificación de canal por lotes ----------
    if codigo == "convolucional":
        bloques_rx = viterbi_lote(llr)
    else:
        o = 0
        Lsys = llr[:, o:o + K]; o += K
        Lp1 = llr[:, o:o + K]; o += K
        Lp2 = llr[:, o:o + K]; o += K
        Lt1s = llr[:, o:o + 3]; o += 3
        Lt1p = llr[:, o:o + 3]; o += 3
        Lt2s = llr[:, o:o + 3]; o += 3
        Lt2p = llr[:, o:o + 3]; o += 3
        bloques_rx = turbo_lote(Lsys, Lp1, Lp2, Lt1s, Lt1p, Lt2s, Lt2p, perm, K)

    # ---------- Reensamblado ----------
    info_rx = bloques_rx.reshape(-1)[:len(bits_tx)]
    salida = {
        "bits_rx": info_rx,
        "ber": calcular_ber(bits_tx, info_rx),
        "n_simbolos_ofdm": n_ofdm,
        "n_bloques": int(n_bloques),
    }
    if capturar_constelaciones:
        salida["constelacion_rx"] = s_eq[:min(len(s_eq), 4000)]
    return salida


# =====================================================================================
# === [BLOQUE 7] ENDPOINTS FLASK — exponen la lógica a la interfaz web               ===
# =====================================================================================

@app.route("/")
def index():
    """Sirve la página principal (interfaz con las pestañas OFDM y SC-FDM)."""
    return render_template("index.html")


@app.route("/subir_imagen", methods=["POST"])
def subir_imagen():
    """Recibe una imagen, la convierte en bits y la guarda en ESTADO (la comparten ambas pestañas)."""
    if "imagen" not in request.files:                              # Verifica que llegó el archivo
        return jsonify({"error": "No se envió imagen"}), 400
    f = request.files["imagen"]
    try:
        img = Image.open(f.stream)                                 # Abre la imagen recibida
        img.load()
    except Exception as e:
        return jsonify({"error": f"No se pudo leer la imagen: {e}"}), 400

    if img.mode not in ("RGB", "L"):                              # Normaliza el modo de color
        img = img.convert("RGB")

    bits, modo, size = imagen_a_bits(img)                         # Convierte la imagen a bits
    ESTADO["bits"] = bits                                         # Guarda el estado de la imagen
    ESTADO["imagen_mode"] = modo
    ESTADO["imagen_size"] = size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    ESTADO["imagen_bytes"] = buf.getvalue()

    return jsonify({                                             # Devuelve metadatos + vista previa
        "n_bits": int(len(bits)),
        "ancho": size[0],
        "alto": size[1],
        "canales": 3 if modo == "RGB" else 1,
        "preview_b64": "data:image/png;base64," + base64.b64encode(ESTADO["imagen_bytes"]).decode("ascii"),
    })


@app.route("/calcular_parametros", methods=["POST"])
def calcular_parametros():
    """Calcula y devuelve los parámetros derivados (N_SC, N_FFT, fs, N_CP, ...) de una configuración."""
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])                                # Ancho de banda
        df = float(data["delta_f_khz"])                          # Espaciamiento entre subportadoras
        cp = data["tipo_cp"]                                     # Tipo de prefijo cíclico
        modulacion = data.get("modulacion", "16-QAM")           # Modulación (para bits/símbolo)
        bps = MODULACIONES[modulacion]["bits"]
        n_bits = int(data.get("n_bits", 0)) or (len(ESTADO["bits"]) if ESTADO["bits"] is not None else 0)
        params = calcular_parametros_ofdm(bw, df, cp, n_bits, bps)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(params)


@app.route("/simular", methods=["POST"])
def simular():
    """
    Ejecuta UNA transmisión completa de la imagen cargada con el esquema indicado
    (OFDM o SC-FDM) y devuelve imagen recuperada, BER, constelaciones, mapa de
    subportadoras y la comparación de forma de onda |x(t)|.
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        snr = float(data["snr_db"])
        modulacion = data["modulacion"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    # --- Selección y validación del esquema (OFDM / SC-FDM) ---
    esquema = data.get("esquema", "OFDM")                        # Esquema solicitado (por defecto OFDM)
    if esquema not in ESQUEMAS:                                  # Valida que exista
        return jsonify({"error": f"Esquema no válido: {esquema}"}), 400
    if modulacion not in ESQUEMAS[esquema]["mods"]:             # Valida que la modulación sea válida para el esquema
        return jsonify({"error": f"{modulacion} no está disponible en {esquema}"}), 400
    usar_dft = ESQUEMAS[esquema]["dft"]                         # True si es SC-FDM, False si es OFDM

    if ESTADO["bits"] is None:                                   # Debe haber una imagen cargada
        return jsonify({"error": "Primero suba una imagen"}), 400

    bps = MODULACIONES[modulacion]["bits"]
    bits_tx = ESTADO["bits"]                                     # Bits de la imagen a transmitir
    params = calcular_parametros_ofdm(bw, df, cp, len(bits_tx), bps)  # Parámetros de la cadena

    rng = np.random.default_rng()                               # Generador aleatorio (canal/ruido)
    t0 = time.perf_counter()
    resultado = cadena_tx_rx(bits_tx, modulacion, params, snr, rng,   # Ejecuta la cadena completa
                              capturar_constelaciones=True, usar_dft_spread=usar_dft)
    tiempo_computo_s = time.perf_counter() - t0

    # --- Comparación de forma de onda |x(t)| OFDM vs SC-FDM ---
    # Se usa un símbolo con datos ALEATORIOS (los mismos para ambos esquemas), no el bloque
    # de la imagen: la imagen es muy estructurada y daría un PAPR atípico. Esto es coherente
    # con la metodología de la CCDF del PAPR.
    n_bits_bloque = params["n_datos"] * bps
    bits_bloque = rng.integers(0, 2, n_bits_bloque, dtype=np.uint8)
    simbolos_bloque = MODULACIONES[modulacion]["mapear"](bits_bloque)
    env_ofdm, papr_ofdm = forma_onda_simbolo(simbolos_bloque, params, usar_dft_spread=False)
    env_scfdm, papr_scfdm = forma_onda_simbolo(simbolos_bloque, params, usar_dft_spread=True)

    # Tiempo "real" de transmisión por el aire: nº de símbolos × duración de cada símbolo
    tiempo_aire_s = resultado["n_simbolos_ofdm"] * params["duracion_simbolo_us"] * 1e-6

    # Reconstruye la imagen recibida y recupera la original
    img_rx = bits_a_imagen(resultado["bits_rx"], ESTADO["imagen_mode"], ESTADO["imagen_size"])
    img_tx = Image.open(io.BytesIO(ESTADO["imagen_bytes"]))

    # Submuestrea las constelaciones para no enviar megabytes al navegador
    def submuestrear(arr, max_n=3000):
        if len(arr) > max_n:
            idx = np.random.choice(len(arr), max_n, replace=False)
            return arr[idx]
        return arr

    c_tx = submuestrear(resultado["constelacion_tx"])
    c_rx = submuestrear(resultado["constelacion_rx"])

    # --- Mapa de subportadoras según el esquema ---
    #  - OFDM   : datos + pilotos intercalados (cada 12).
    #  - SC-FDM : bloque CONTIGUO de datos (localizado) + subportadoras de guarda; sin pilotos.
    if usar_dft:
        datos_idx = np.arange(params["n_datos"])                # Datos = bloque contiguo
        pilotos_idx = np.array([], dtype=int)                   # SC-FDM no intercala pilotos
        guarda_idx = np.arange(params["n_datos"], params["n_sc"])  # Resto = guarda
    else:
        pilotos_idx, datos_idx = indices_pilotos_datos(params["n_sc"])  # OFDM: pilotos + datos
        guarda_idx = np.array([], dtype=int)

    return jsonify({                                            # Respuesta JSON al frontend
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


@app.route("/montecarlo", methods=["POST"])
def montecarlo():
    """
    Simulación de Monte Carlo. Acepta una lista `esquemas` y devuelve series planas:
      - esquemas=["OFDM"]            -> 3 curvas (Práctica 1, pestaña OFDM).
      - esquemas=["OFDM","SC-FDM"]   -> 5 curvas combinadas (Práctica 2, pestaña SC-FDM).
    Cada serie lleva su esquema y modulación para que el frontend la dibuje (color =
    modulación, estilo de línea = esquema). Calcula BER vs SNR (con IC 95%) y la CCDF del PAPR.
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    # Esquemas a simular (se filtran contra ESQUEMAS para evitar valores inválidos)
    esquemas = [e for e in data.get("esquemas", ["OFDM"]) if e in ESQUEMAS]
    if not esquemas:
        esquemas = ["OFDM"]
    combinado = len(esquemas) > 1                       # ¿Es la comparación combinada (5 curvas)?

    # En modo combinado se reduce la carga (5×16×10 corridas) para que la simulación sea ágil
    N_BITS_MC = 20000 if combinado else 50000          # Bits por corrida
    N_SIMBOLOS_PAPR = 1000 if combinado else 1500      # Símbolos para la CCDF del PAPR
    snr_valores = list(range(0, 16))                   # Barrido de SNR de 0 a 15 dB
    n_sim = 10                                         # Corridas independientes por punto
    rng = np.random.default_rng()

    # Valor crítico de la t de Student al 95% con n-1 = 9 grados de libertad
    t_critico = float(t_student.ppf(0.975, df=n_sim - 1))

    # --- BER vs SNR: una serie por cada par esquema×modulación ---
    series_ber = []
    for esquema in esquemas:                            # Recorre OFDM y, si aplica, SC-FDM
        usar_dft = ESQUEMAS[esquema]["dft"]            # Selecciona la rama de la cadena
        for modulacion in ESQUEMAS[esquema]["mods"]:   # Recorre las modulaciones del esquema
            bps = MODULACIONES[modulacion]["bits"]
            params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)
            ber_prom, ic_inf, ic_sup = [], [], []      # Acumuladores de la curva
            for snr in snr_valores:                    # Barre cada valor de SNR
                bers = []
                for _ in range(n_sim):                 # n_sim corridas independientes
                    bits_aleatorios = rng.integers(0, 2, N_BITS_MC, dtype=np.uint8)
                    r = cadena_tx_rx(bits_aleatorios, modulacion, params, snr, rng,
                                      capturar_constelaciones=False, usar_dft_spread=usar_dft)
                    bers.append(r["ber"])
                bers = np.array(bers)
                mu = float(np.mean(bers))              # BER promedio del punto
                sd = float(np.std(bers, ddof=1)) if n_sim > 1 else 0.0  # Desv. estándar muestral
                margen = t_critico * sd / np.sqrt(n_sim)  # Margen del intervalo de confianza
                ber_prom.append(mu)
                ic_inf.append(max(mu - margen, 1e-6))  # Límite inferior (acotado para escala log)
                ic_sup.append(mu + margen)             # Límite superior
            series_ber.append({                        # Guarda la serie completa
                "esquema": esquema,
                "modulacion": modulacion,
                "ber_promedio": ber_prom,
                "ic_inferior": ic_inf,
                "ic_superior": ic_sup,
            })

    # --- CCDF del PAPR (independiente del SNR), sobre un eje común a todas las curvas ---
    eje_comun = np.linspace(0, 15, 80)                 # Eje x compartido (necesario para superponer)
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

    return jsonify({                                   # Respuesta con todas las series
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


@app.route("/simular_mrc", methods=["POST"])
def simular_mrc():
    """
    Práctica 3 (diversidad en RX): ejecuta UNA transmisión de la imagen cargada con el
    transmisor OFDM y un receptor de n_rx antenas que combina con MRC. Devuelve la imagen
    recuperada, la BER y las constelaciones RX antes (1 antena, ZF) y después de MRC.
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

    if modulacion not in ESQUEMAS["OFDM"]["mods"]:                   # El transmisor de la práctica es OFDM
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400
    if n_rx < 1:
        return jsonify({"error": "n_rx debe ser >= 1"}), 400
    if ESTADO["bits"] is None:                                       # Debe haber una imagen cargada
        return jsonify({"error": "Primero suba una imagen"}), 400

    bps = MODULACIONES[modulacion]["bits"]
    bits_tx = ESTADO["bits"]                                         # Bits de la imagen (mismo TX)
    params = calcular_parametros_ofdm(bw, df, cp, len(bits_tx), bps)

    rng = np.random.default_rng()
    t0 = time.perf_counter()
    resultado = cadena_tx_rx(bits_tx, modulacion, params, snr, rng,  # Cadena con receptor MRC
                              capturar_constelaciones=True, usar_dft_spread=False, n_rx=n_rx)
    tiempo_computo_s = time.perf_counter() - t0

    tiempo_aire_s = resultado["n_simbolos_ofdm"] * params["duracion_simbolo_us"] * 1e-6
    img_rx = bits_a_imagen(resultado["bits_rx"], ESTADO["imagen_mode"], ESTADO["imagen_size"])
    img_tx = Image.open(io.BytesIO(ESTADO["imagen_bytes"]))

    def submuestrear(arr, max_n=3000):                              # Limita los puntos enviados al navegador
        if len(arr) > max_n:
            idx = np.random.choice(len(arr), max_n, replace=False)
            return arr[idx]
        return arr

    c_rx = submuestrear(resultado["constelacion_rx"])
    salida = {                                                      # Respuesta JSON al frontend
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
    Práctica 3: Monte Carlo de BER vs SNR para diversidad en RX con MRC. Para la modulación
    seleccionada calcula una curva por cada nº de antenas en `antenas` (def. [2, 4, 8]). Al
    aumentar el nº de antenas la curva debe bajar (ganancia de diversidad). IC 95% t-Student.
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        modulacion = data["modulacion"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    if modulacion not in ESQUEMAS["OFDM"]["mods"]:
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400

    # Lista de nº de antenas a simular (se filtran valores válidos >= 1)
    antenas = [int(a) for a in data.get("antenas", [2, 4, 8]) if int(a) >= 1]
    if not antenas:
        antenas = [2, 4, 8]

    N_BITS_MC = 10000                                  # Bits por corrida (reducido por la carga de MRC)
    snr_valores = list(range(0, 16))                   # Barrido de SNR de 0 a 15 dB
    n_sim = 8                                          # Corridas independientes por punto
    rng = np.random.default_rng()
    # Valor crítico de la t de Student al 95% con n-1 grados de libertad
    t_critico = float(t_student.ppf(0.975, df=n_sim - 1))

    bps = MODULACIONES[modulacion]["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)

    series_ber = []
    for n_rx in antenas:                               # Una curva BER vs SNR por cada nº de antenas
        ber_prom, ic_inf, ic_sup = [], [], []
        for snr in snr_valores:                        # Barre cada valor de SNR
            bers = []
            for _ in range(n_sim):                     # n_sim corridas independientes
                bits_aleatorios = rng.integers(0, 2, N_BITS_MC, dtype=np.uint8)
                r = cadena_tx_rx(bits_aleatorios, modulacion, params, snr, rng,
                                  capturar_constelaciones=False, usar_dft_spread=False, n_rx=n_rx)
                bers.append(r["ber"])
            bers = np.array(bers)
            mu = float(np.mean(bers))                  # BER promedio del punto
            sd = float(np.std(bers, ddof=1)) if n_sim > 1 else 0.0
            margen = t_critico * sd / np.sqrt(n_sim)   # Margen del intervalo de confianza
            ber_prom.append(mu)
            ic_inf.append(max(mu - margen, 1e-6))      # Límite inferior (acotado para escala log)
            ic_sup.append(mu + margen)                 # Límite superior
        series_ber.append({                            # Guarda la curva de este nº de antenas
            "n_rx": n_rx,
            "ber_promedio": ber_prom,
            "ic_inferior": ic_inf,
            "ic_superior": ic_sup,
        })

    return jsonify({                                   # Respuesta con todas las curvas
        "snr_valores": snr_valores,
        "n_simulaciones": n_sim,
        "t_critico": t_critico,
        "n_bits_mc": N_BITS_MC,
        "modulacion": modulacion,
        "series_ber": series_ber,
    })


@app.route("/simular_sfbc", methods=["POST"])
def simular_sfbc():
    """
    Práctica 4 (diversidad en TX): ejecuta UNA transmisión de la imagen cargada con SFBC
    (Alamouti) en la configuración elegida (2x1 o 2x2). Para la comparación "antes vs después"
    ejecuta además una cadena SISO (1x1, sin diversidad) sobre los mismos bits. Devuelve la imagen
    recuperada, la BER y las constelaciones RX sin diversidad (antes) y con SFBC (después).
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        snr = float(data["snr_db"])
        modulacion = data["modulacion"]
        config = data["config"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    CONFIGS = {"2x1": (2, 1), "2x2": (2, 2)}                          # Configuraciones SFBC válidas
    if modulacion not in ESQUEMAS["OFDM"]["mods"]:                   # El transmisor de la práctica es OFDM
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400
    if config not in CONFIGS:
        return jsonify({"error": f"Configuración '{config}' no válida (use 2x1 o 2x2)"}), 400
    if ESTADO["bits"] is None:                                       # Debe haber una imagen cargada
        return jsonify({"error": "Primero suba una imagen"}), 400

    n_tx, n_rx = CONFIGS[config]
    bps = MODULACIONES[modulacion]["bits"]
    bits_tx = ESTADO["bits"]                                         # Bits de la imagen (mismo TX)
    params = calcular_parametros_ofdm(bw, df, cp, len(bits_tx), bps)

    rng = np.random.default_rng()
    t0 = time.perf_counter()
    resultado = cadena_tx_sfbc_rx(bits_tx, modulacion, params, snr, rng,   # "Después": SFBC elegido
                                   capturar_constelaciones=True, n_tx=n_tx, n_rx=n_rx)
    ref = cadena_tx_sfbc_rx(bits_tx, modulacion, params, snr, rng,         # "Antes": SISO 1x1 de referencia
                            capturar_constelaciones=True, n_tx=1, n_rx=1)
    tiempo_computo_s = time.perf_counter() - t0

    tiempo_aire_s = resultado["n_simbolos_ofdm"] * params["duracion_simbolo_us"] * 1e-6
    img_rx = bits_a_imagen(resultado["bits_rx"], ESTADO["imagen_mode"], ESTADO["imagen_size"])
    img_tx = Image.open(io.BytesIO(ESTADO["imagen_bytes"]))

    def submuestrear(arr, max_n=3000):                              # Limita los puntos enviados al navegador
        if len(arr) > max_n:
            idx = np.random.choice(len(arr), max_n, replace=False)
            return arr[idx]
        return arr

    c_rx = submuestrear(resultado["constelacion_rx"])
    c_antes = submuestrear(ref["constelacion_rx"])
    salida = {                                                      # Respuesta JSON al frontend
        "ber": resultado["ber"],
        "ber_antes": ref["ber"],
        "modulacion": modulacion,
        "config": config,
        "n_tx": n_tx,
        "n_rx": n_rx,
        "snr_db": snr,
        "bits_transmitidos": int(len(bits_tx)),
        "bits_erroneos": int(round(resultado["ber"] * len(bits_tx))),
        "n_simbolos_ofdm": int(resultado["n_simbolos_ofdm"]),
        "imagen_original_b64": img_a_b64(img_tx),
        "imagen_recuperada_b64": img_a_b64(img_rx),
        "constelacion_rx": {"real": c_rx.real.tolist(), "imag": c_rx.imag.tolist()},
        "constelacion_rx_antes": {"real": c_antes.real.tolist(), "imag": c_antes.imag.tolist()},
        "parametros": params,
        "tiempo_aire_s": tiempo_aire_s,
        "tiempo_computo_s": tiempo_computo_s,
    }
    return jsonify(salida)


@app.route("/montecarlo_sfbc", methods=["POST"])
def montecarlo_sfbc():
    """
    Práctica 4: Monte Carlo de BER vs SNR para diversidad en TX. Calcula una curva por cada
    configuración en `configs` (def. ["1x1", "2x1", "2x2"]). Al subir el orden de diversidad
    (1 -> 2 -> 4) la curva debe caer con mayor pendiente. IC 95% con la t de Student.
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        modulacion = data["modulacion"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    if modulacion not in ESQUEMAS["OFDM"]["mods"]:
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400

    # Configuraciones a simular: nombre -> (n_tx, n_rx). Diversidad = n_tx·n_rx aquí: 1, 2, 4.
    CONFIGS = {"1x1": (1, 1), "2x1": (2, 1), "2x2": (2, 2)}
    configs = [c for c in data.get("configs", ["1x1", "2x1", "2x2"]) if c in CONFIGS]
    if not configs:
        configs = ["1x1", "2x1", "2x2"]

    N_BITS_MC = 10000                                  # Bits por corrida (igual que la práctica MRC)
    snr_valores = list(range(0, 16))                   # Barrido de SNR de 0 a 15 dB
    n_sim = 8                                          # Corridas independientes por punto
    rng = np.random.default_rng()
    # Valor crítico de la t de Student al 95% con n-1 grados de libertad
    t_critico = float(t_student.ppf(0.975, df=n_sim - 1))

    bps = MODULACIONES[modulacion]["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)

    series_ber = []
    for cfg in configs:                                # Una curva BER vs SNR por configuración
        n_tx, n_rx = CONFIGS[cfg]
        ber_prom, ic_inf, ic_sup = [], [], []
        for snr in snr_valores:                        # Barre cada valor de SNR
            bers = []
            for _ in range(n_sim):                     # n_sim corridas independientes
                bits_aleatorios = rng.integers(0, 2, N_BITS_MC, dtype=np.uint8)
                r = cadena_tx_sfbc_rx(bits_aleatorios, modulacion, params, snr, rng,
                                       capturar_constelaciones=False, n_tx=n_tx, n_rx=n_rx)
                bers.append(r["ber"])
            bers = np.array(bers)
            mu = float(np.mean(bers))                  # BER promedio del punto
            sd = float(np.std(bers, ddof=1)) if n_sim > 1 else 0.0
            margen = t_critico * sd / np.sqrt(n_sim)   # Margen del intervalo de confianza
            ber_prom.append(mu)
            ic_inf.append(max(mu - margen, 1e-6))      # Límite inferior (acotado para escala log)
            ic_sup.append(mu + margen)                 # Límite superior
        series_ber.append({                            # Guarda la curva de esta configuración
            "config": cfg,
            "ber_promedio": ber_prom,
            "ic_inferior": ic_inf,
            "ic_superior": ic_sup,
        })

    return jsonify({                                   # Respuesta con todas las curvas
        "snr_valores": snr_valores,
        "n_simulaciones": n_sim,
        "t_critico": t_critico,
        "n_bits_mc": N_BITS_MC,
        "modulacion": modulacion,
        "series_ber": series_ber,
    })


@app.route("/simular_beamforming", methods=["POST"])
def simular_beamforming():
    """
    Práctica 5 (beamforming en TX con MRT): ejecuta UNA transmisión de la imagen cargada con
    beamforming en la configuración elegida (2x1 o 4x1). Para la comparación "antes vs después"
    ejecuta además una cadena SISO (1x1, sin beamforming) sobre los mismos bits. Devuelve la
    imagen recuperada, la BER y las constelaciones RX sin beamforming (antes) y con MRT (después).
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        snr = float(data["snr_db"])
        modulacion = data["modulacion"]
        config = data["config"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    CONFIGS = {"2x1": 2, "4x1": 4}                                    # Configuraciones BF válidas (n_tx)
    if modulacion not in ESQUEMAS["OFDM"]["mods"]:                   # El transmisor de la práctica es OFDM
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400
    if config not in CONFIGS:
        return jsonify({"error": f"Configuración '{config}' no válida (use 2x1 o 4x1)"}), 400
    if ESTADO["bits"] is None:                                       # Debe haber una imagen cargada
        return jsonify({"error": "Primero suba una imagen"}), 400

    n_tx = CONFIGS[config]
    bps = MODULACIONES[modulacion]["bits"]
    bits_tx = ESTADO["bits"]                                         # Bits de la imagen (mismo TX)
    params = calcular_parametros_ofdm(bw, df, cp, len(bits_tx), bps)

    rng = np.random.default_rng()
    t0 = time.perf_counter()
    resultado = cadena_tx_beamforming_rx(bits_tx, modulacion, params, snr, rng,  # "Después": BF elegido
                                         capturar_constelaciones=True, n_tx=n_tx)
    ref = cadena_tx_beamforming_rx(bits_tx, modulacion, params, snr, rng,        # "Antes": SISO 1x1
                                   capturar_constelaciones=True, n_tx=1)
    tiempo_computo_s = time.perf_counter() - t0

    tiempo_aire_s = resultado["n_simbolos_ofdm"] * params["duracion_simbolo_us"] * 1e-6
    img_rx = bits_a_imagen(resultado["bits_rx"], ESTADO["imagen_mode"], ESTADO["imagen_size"])
    img_tx = Image.open(io.BytesIO(ESTADO["imagen_bytes"]))

    def submuestrear(arr, max_n=3000):                              # Limita los puntos enviados al navegador
        if len(arr) > max_n:
            idx = np.random.choice(len(arr), max_n, replace=False)
            return arr[idx]
        return arr

    c_rx = submuestrear(resultado["constelacion_rx"])
    c_antes = submuestrear(ref["constelacion_rx"])
    salida = {                                                      # Respuesta JSON al frontend
        "ber": resultado["ber"],
        "ber_antes": ref["ber"],
        "modulacion": modulacion,
        "config": config,
        "n_tx": n_tx,
        "n_rx": 1,
        "snr_db": snr,
        "bits_transmitidos": int(len(bits_tx)),
        "bits_erroneos": int(round(resultado["ber"] * len(bits_tx))),
        "n_simbolos_ofdm": int(resultado["n_simbolos_ofdm"]),
        "imagen_original_b64": img_a_b64(img_tx),
        "imagen_recuperada_b64": img_a_b64(img_rx),
        "constelacion_rx": {"real": c_rx.real.tolist(), "imag": c_rx.imag.tolist()},
        "constelacion_rx_antes": {"real": c_antes.real.tolist(), "imag": c_antes.imag.tolist()},
        "parametros": params,
        "tiempo_aire_s": tiempo_aire_s,
        "tiempo_computo_s": tiempo_computo_s,
    }
    return jsonify(salida)


def _curva_ber_mc(funcion_corrida, snr_valores, n_sim, t_critico):
    """
    Apoyo del Monte Carlo: para cada SNR ejecuta n_sim corridas de `funcion_corrida(snr)` (que
    devuelve una BER) y arma la curva con su intervalo de confianza al 95% (t de Student).
    """
    ber_prom, ic_inf, ic_sup = [], [], []
    for snr in snr_valores:
        bers = np.array([funcion_corrida(snr) for _ in range(n_sim)])
        mu = float(np.mean(bers))
        sd = float(np.std(bers, ddof=1)) if n_sim > 1 else 0.0
        margen = t_critico * sd / np.sqrt(n_sim)
        ber_prom.append(mu)
        ic_inf.append(max(mu - margen, 1e-6))
        ic_sup.append(mu + margen)
    return ber_prom, ic_inf, ic_sup


@app.route("/montecarlo_beamforming", methods=["POST"])
def montecarlo_beamforming():
    """
    Práctica 5: Monte Carlo de BER vs SNR para beamforming en TX (MRT). Devuelve DOS conjuntos
    de curvas:
      - series_bf: órdenes de beamforming 1×1 / 2×1 / 4×1. Al subir n_tx la curva baja con mayor
        pendiente (diversidad de orden n_tx) y se corre a la izquierda (ganancia de arreglo).
      - series_overlay: comparación de las 3 prácticas al MISMO orden de diversidad 2:
        BF 2×1 (MRT, P5) vs SFBC 2×1 (Alamouti, P4) vs MRC 1×2 (P3). BF y MRC se solapan (son
        duales: ~3 dB de ganancia de arreglo + orden 2) y le ganan ~3 dB a SFBC, que reparte la
        potencia a ciegas por no tener CSI en el TX. IC 95% con la t de Student.
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        modulacion = data["modulacion"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    if modulacion not in ESQUEMAS["OFDM"]["mods"]:
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400

    N_BITS_MC = 10000                                  # Bits por corrida (igual que MRC/SFBC)
    snr_valores = list(range(0, 16))                   # Barrido de SNR de 0 a 15 dB
    n_sim = 8                                          # Corridas independientes por punto
    rng = np.random.default_rng()
    t_critico = float(t_student.ppf(0.975, df=n_sim - 1))   # t de Student al 95%

    bps = MODULACIONES[modulacion]["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)

    def corrida(funcion):                              # Genera bits nuevos y devuelve la BER de una corrida
        return lambda snr: funcion(rng.integers(0, 2, N_BITS_MC, dtype=np.uint8), snr)

    # --- Órdenes de beamforming: 1×1 / 2×1 / 4×1 ---
    CONFIGS_BF = [("1x1", 1), ("2x1", 2), ("4x1", 4)]
    series_bf, curva_bf_2x1 = [], None
    for nombre, n_tx in CONFIGS_BF:
        f = corrida(lambda bits, snr, n_tx=n_tx: cadena_tx_beamforming_rx(
            bits, modulacion, params, snr, rng, n_tx=n_tx)["ber"])
        ber_prom, ic_inf, ic_sup = _curva_ber_mc(f, snr_valores, n_sim, t_critico)
        item = {"config": nombre, "ber_promedio": ber_prom, "ic_inferior": ic_inf, "ic_superior": ic_sup}
        series_bf.append(item)
        if nombre == "2x1":
            curva_bf_2x1 = item

    # --- Overlay comparativo (mismo orden de diversidad 2): BF 2×1 / SFBC 2×1 / MRC 1×2 ---
    # Las tres se calculan con el MISMO piso de ruido fijo (ruido_piso_fijo=True) que el beamforming,
    # para que la comparación sea justa: BF 2×1 y MRC 1×2 se solapan (duales) y le ganan ~3 dB a SFBC.
    f_sfbc = corrida(lambda bits, snr: cadena_tx_sfbc_rx(
        bits, modulacion, params, snr, rng, n_tx=2, n_rx=1, ruido_piso_fijo=True)["ber"])
    sfbc_prom, sfbc_inf, sfbc_sup = _curva_ber_mc(f_sfbc, snr_valores, n_sim, t_critico)
    f_mrc = corrida(lambda bits, snr: cadena_tx_rx(
        bits, modulacion, params, snr, rng, usar_dft_spread=False, n_rx=2, ruido_piso_fijo=True)["ber"])
    mrc_prom, mrc_inf, mrc_sup = _curva_ber_mc(f_mrc, snr_valores, n_sim, t_critico)
    series_overlay = [
        {"tecnica": "BF 2x1", "ber_promedio": curva_bf_2x1["ber_promedio"],
         "ic_inferior": curva_bf_2x1["ic_inferior"], "ic_superior": curva_bf_2x1["ic_superior"]},
        {"tecnica": "SFBC 2x1", "ber_promedio": sfbc_prom, "ic_inferior": sfbc_inf, "ic_superior": sfbc_sup},
        {"tecnica": "MRC 1x2", "ber_promedio": mrc_prom, "ic_inferior": mrc_inf, "ic_superior": mrc_sup},
    ]

    return jsonify({                                   # Respuesta con los dos conjuntos de curvas
        "snr_valores": snr_valores,
        "n_simulaciones": n_sim,
        "t_critico": t_critico,
        "n_bits_mc": N_BITS_MC,
        "modulacion": modulacion,
        "series_bf": series_bf,
        "series_overlay": series_overlay,
    })


@app.route("/simular_mimo", methods=["POST"])
def simular_mimo():
    """
    Práctica 6 (diversidad combinada TX+RX): ejecuta UNA transmisión de la imagen cargada con
    SFBC (Alamouti, 2 TX) + MRC (n_rx RX) en la configuración elegida (2x1 / 2x2 / 2x4). Para
    la comparación "antes vs después" ejecuta además una cadena SISO 1x1 (sin diversidad) sobre
    los mismos bits. Devuelve la imagen recuperada, la BER, la orden de diversidad (2·n_rx) y las
    constelaciones RX sin diversidad (antes) y con SFBC+MRC (después).
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        snr = float(data["snr_db"])
        modulacion = data["modulacion"]
        config = data["config"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    CONFIGS = {"2x1": (2, 1), "2x2": (2, 2), "2x4": (2, 4)}           # Configuraciones SFBC+MRC válidas
    if modulacion not in ESQUEMAS["OFDM"]["mods"]:                   # El transmisor de la práctica es OFDM
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400
    if config not in CONFIGS:
        return jsonify({"error": f"Configuración '{config}' no válida (use 2x1, 2x2 o 2x4)"}), 400
    if ESTADO["bits"] is None:                                       # Debe haber una imagen cargada
        return jsonify({"error": "Primero suba una imagen"}), 400

    n_tx, n_rx = CONFIGS[config]
    bps = MODULACIONES[modulacion]["bits"]
    bits_tx = ESTADO["bits"]                                         # Bits de la imagen (mismo TX)
    params = calcular_parametros_ofdm(bw, df, cp, len(bits_tx), bps)

    rng = np.random.default_rng()
    t0 = time.perf_counter()
    resultado = cadena_tx_mimo_div_rx(bits_tx, modulacion, params, snr, rng,   # "Después": SFBC+MRC elegido
                                      capturar_constelaciones=True, n_tx=n_tx, n_rx=n_rx)
    ref = cadena_tx_mimo_div_rx(bits_tx, modulacion, params, snr, rng,         # "Antes": SISO 1x1 sin diversidad
                                capturar_constelaciones=True, n_tx=1, n_rx=1)
    tiempo_computo_s = time.perf_counter() - t0

    tiempo_aire_s = resultado["n_simbolos_ofdm"] * params["duracion_simbolo_us"] * 1e-6
    img_rx = bits_a_imagen(resultado["bits_rx"], ESTADO["imagen_mode"], ESTADO["imagen_size"])
    img_tx = Image.open(io.BytesIO(ESTADO["imagen_bytes"]))

    def submuestrear(arr, max_n=3000):                              # Limita los puntos enviados al navegador
        if len(arr) > max_n:
            idx = np.random.choice(len(arr), max_n, replace=False)
            return arr[idx]
        return arr

    c_rx = submuestrear(resultado["constelacion_rx"])
    c_antes = submuestrear(ref["constelacion_rx"])
    salida = {                                                      # Respuesta JSON al frontend
        "ber": resultado["ber"],
        "ber_antes": ref["ber"],
        "modulacion": modulacion,
        "config": config,
        "n_tx": n_tx,
        "n_rx": n_rx,
        "orden_diversidad": n_tx * n_rx,
        "snr_db": snr,
        "bits_transmitidos": int(len(bits_tx)),
        "bits_erroneos": int(round(resultado["ber"] * len(bits_tx))),
        "n_simbolos_ofdm": int(resultado["n_simbolos_ofdm"]),
        "imagen_original_b64": img_a_b64(img_tx),
        "imagen_recuperada_b64": img_a_b64(img_rx),
        "constelacion_rx": {"real": c_rx.real.tolist(), "imag": c_rx.imag.tolist()},
        "constelacion_rx_antes": {"real": c_antes.real.tolist(), "imag": c_antes.imag.tolist()},
        "parametros": params,
        "tiempo_aire_s": tiempo_aire_s,
        "tiempo_computo_s": tiempo_computo_s,
    }
    return jsonify(salida)


@app.route("/montecarlo_mimo", methods=["POST"])
def montecarlo_mimo():
    """
    Práctica 6: Monte Carlo de BER vs SNR para diversidad combinada TX+RX. Devuelve DOS conjuntos
    de curvas:
      - series_mimo: la combinación SFBC+MRC a distintos tamaños 2x1 / 2x2 / 2x4 (orden de
        diversidad 2 / 4 / 8). Al subir n_rx la curva cae con MAYOR PENDIENTE.
      - series_overlay: de dónde viene la diversidad, al mismo esfuerzo — SISO 1x1 (orden 1),
        SFBC 2x1 (solo TX, orden 2), MRC 1x2 (solo RX, orden 2) y SFBC+MRC 2x2 (combinado,
        orden 4). Muestra que combinar TX y RX MULTIPLICA la orden (2×2=4): la mezcla cae más
        rápido que cualquiera de las diversidades por separado. Todas con la MISMA calibración de
        ruido (al recibido). IC 95% con la t de Student.
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        modulacion = data["modulacion"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    if modulacion not in ESQUEMAS["OFDM"]["mods"]:
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400

    N_BITS_MC = 10000                                  # Bits por corrida (igual que MRC/SFBC/BF)
    snr_valores = list(range(0, 16))                   # Barrido de SNR de 0 a 15 dB
    n_sim = 8                                          # Corridas independientes por punto
    rng = np.random.default_rng()
    t_critico = float(t_student.ppf(0.975, df=n_sim - 1))   # t de Student al 95%

    bps = MODULACIONES[modulacion]["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)

    def corrida(funcion):                              # Genera bits nuevos y devuelve la BER de una corrida
        return lambda snr: funcion(rng.integers(0, 2, N_BITS_MC, dtype=np.uint8), snr)

    # --- La mezcla SFBC+MRC a distintos tamaños: 2x1 / 2x2 / 2x4 (orden 2 / 4 / 8) ---
    CONFIGS = [("2x1", 2, 1), ("2x2", 2, 2), ("2x4", 2, 4)]
    series_mimo, curva_2x2 = [], None
    for nombre, n_tx, n_rx in CONFIGS:
        f = corrida(lambda bits, snr, n_tx=n_tx, n_rx=n_rx: cadena_tx_mimo_div_rx(
            bits, modulacion, params, snr, rng, n_tx=n_tx, n_rx=n_rx)["ber"])
        ber_prom, ic_inf, ic_sup = _curva_ber_mc(f, snr_valores, n_sim, t_critico)
        item = {"config": nombre, "orden": n_tx * n_rx,
                "ber_promedio": ber_prom, "ic_inferior": ic_inf, "ic_superior": ic_sup}
        series_mimo.append(item)
        if nombre == "2x2":
            curva_2x2 = item

    # --- Overlay: de dónde viene la diversidad (SISO / solo TX / solo RX / combinado) ---
    f_siso = corrida(lambda bits, snr: cadena_tx_rx(
        bits, modulacion, params, snr, rng, usar_dft_spread=False, n_rx=1)["ber"])
    siso_prom, siso_inf, siso_sup = _curva_ber_mc(f_siso, snr_valores, n_sim, t_critico)
    f_sfbc = corrida(lambda bits, snr: cadena_tx_sfbc_rx(
        bits, modulacion, params, snr, rng, n_tx=2, n_rx=1)["ber"])
    sfbc_prom, sfbc_inf, sfbc_sup = _curva_ber_mc(f_sfbc, snr_valores, n_sim, t_critico)
    f_mrc = corrida(lambda bits, snr: cadena_tx_rx(
        bits, modulacion, params, snr, rng, usar_dft_spread=False, n_rx=2)["ber"])
    mrc_prom, mrc_inf, mrc_sup = _curva_ber_mc(f_mrc, snr_valores, n_sim, t_critico)
    series_overlay = [
        {"tecnica": "SISO 1x1", "ber_promedio": siso_prom, "ic_inferior": siso_inf, "ic_superior": siso_sup},
        {"tecnica": "SFBC 2x1", "ber_promedio": sfbc_prom, "ic_inferior": sfbc_inf, "ic_superior": sfbc_sup},
        {"tecnica": "MRC 1x2", "ber_promedio": mrc_prom, "ic_inferior": mrc_inf, "ic_superior": mrc_sup},
        {"tecnica": "SFBC+MRC 2x2", "ber_promedio": curva_2x2["ber_promedio"],
         "ic_inferior": curva_2x2["ic_inferior"], "ic_superior": curva_2x2["ic_superior"]},
    ]

    return jsonify({                                   # Respuesta con los dos conjuntos de curvas
        "snr_valores": snr_valores,
        "n_simulaciones": n_sim,
        "t_critico": t_critico,
        "n_bits_mc": N_BITS_MC,
        "modulacion": modulacion,
        "series_mimo": series_mimo,
        "series_overlay": series_overlay,
    })


_CODIGOS_CODIF = {"ninguno", "convolucional", "turbo"}
_MODS_CODIF = ["QPSK", "16-QAM"]                       # el demapeo suave (LLR) cubre estas dos


def texto_a_bits(texto):
    """Texto (UTF-8) -> vector de bits (MSB primero)."""
    data = texto.encode("utf-8")
    if len(data) == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_a_texto(bits):
    """Vector de bits -> texto (UTF-8, reemplazando bytes inválidos por �)."""
    n = (len(bits) // 8) * 8
    if n == 0:
        return ""
    return np.packbits(bits[:n].astype(np.uint8)).tobytes().decode("utf-8", errors="replace")


@app.route("/simular_codif", methods=["POST"])
def simular_codif():
    """
    Práctica 7 (codificación de canal): transmite la FUENTE (una imagen cargada o un TEXTO
    simple) DOS veces sobre la misma cadena OFDM + Pedestrian A: con el código elegido
    (convolucional o turbo) y SIN codificar. Devuelve ambas recuperaciones y sus BER, para
    ver la GANANCIA DE CODIFICACIÓN (la versión codificada llega limpia). El texto es mucho
    más corto que una imagen, por lo que la simulación (sobre todo el turbo) es más rápida.
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        snr = float(data["snr_db"])
        modulacion = data["modulacion"]
        codigo = data["codigo"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    if modulacion not in _MODS_CODIF:
        return jsonify({"error": f"{modulacion} no disponible (use QPSK o 16-QAM)"}), 400
    if codigo not in _CODIGOS_CODIF or codigo == "ninguno":
        return jsonify({"error": "Elija un código: 'convolucional' o 'turbo'"}), 400

    # --- Fuente: texto simple o imagen ---
    fuente = data.get("fuente", "imagen")
    if fuente == "texto":
        texto = (data.get("texto") or "").strip()
        if not texto:
            return jsonify({"error": "Escriba un texto para transmitir"}), 400
        bits_tx = texto_a_bits(texto)
    else:
        if ESTADO["bits"] is None:
            return jsonify({"error": "Primero suba una imagen (o use la fuente de texto)"}), 400
        bits_tx = ESTADO["bits"]

    bps = MODULACIONES[modulacion]["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, len(bits_tx), bps)

    rng = np.random.default_rng()
    t0 = time.perf_counter()
    res = cadena_tx_codif_rx(bits_tx, modulacion, codigo, params, snr, rng,    # con código
                             capturar_constelaciones=True)
    tiempo_computo_s = time.perf_counter() - t0           # tiempo de procesar SOLO el código elegido
    ref = cadena_tx_codif_rx(bits_tx, modulacion, "ninguno", params, snr, rng)  # sin codificar (referencia)

    tiempo_aire_s = res["n_simbolos_ofdm"] * params["duracion_simbolo_us"] * 1e-6
    c_rx = res.get("constelacion_rx", np.array([], dtype=complex))
    if len(c_rx) > 3000:
        c_rx = c_rx[np.random.choice(len(c_rx), 3000, replace=False)]

    salida = {
        "fuente": fuente,
        "ber": res["ber"],
        "ber_sin": ref["ber"],
        "modulacion": modulacion,
        "codigo": codigo,
        "snr_db": snr,
        "bits_transmitidos": int(len(bits_tx)),
        "bits_erroneos": int(round(res["ber"] * len(bits_tx))),
        "bits_erroneos_sin": int(round(ref["ber"] * len(bits_tx))),
        "n_bloques": res["n_bloques"],
        "tasa_codigo": "1/3",
        "n_simbolos_ofdm": int(res["n_simbolos_ofdm"]),
        "n_simbolos_ofdm_sin": int(ref["n_simbolos_ofdm"]),
        "constelacion_rx": {"real": c_rx.real.tolist(), "imag": c_rx.imag.tolist()},
        "parametros": params,
        "tiempo_aire_s": tiempo_aire_s,
        "tiempo_computo_s": tiempo_computo_s,
    }
    if fuente == "texto":
        salida["texto_original"] = texto
        salida["texto_recuperado"] = bits_a_texto(res["bits_rx"])
        salida["texto_recuperado_sin"] = bits_a_texto(ref["bits_rx"])
    else:
        img_cod = bits_a_imagen(res["bits_rx"], ESTADO["imagen_mode"], ESTADO["imagen_size"])
        img_sin = bits_a_imagen(ref["bits_rx"], ESTADO["imagen_mode"], ESTADO["imagen_size"])
        img_tx = Image.open(io.BytesIO(ESTADO["imagen_bytes"]))
        salida["imagen_original_b64"] = img_a_b64(img_tx)
        salida["imagen_recuperada_b64"] = img_a_b64(img_cod)
        salida["imagen_recuperada_sin_b64"] = img_a_b64(img_sin)
    return jsonify(salida)


@app.route("/montecarlo_codif", methods=["POST"])
def montecarlo_codif():
    """
    Práctica 7: Monte Carlo de BER vs SNR con TRES curvas sobre la misma cadena OFDM +
    Pedestrian A: sin codificar / convolucional / turbo. Debe verse turbo ≤ convolucional ≤
    sin codificar (ganancia de codificación), con el "acantilado" del turbo. IC 95% (t-Student).
    """
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        modulacion = data["modulacion"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400
    if modulacion not in _MODS_CODIF:
        return jsonify({"error": f"{modulacion} no disponible (use QPSK o 16-QAM)"}), 400

    N_BITS_MC = 3 * PAYLOAD_BLOQUE                      # 3 bloques por corrida
    snr_valores = list(range(0, 16))                   # Barrido de SNR de 0 a 15 dB
    n_sim = 6                                           # Corridas independientes por punto
    rng = np.random.default_rng()
    t_critico = float(t_student.ppf(0.975, df=n_sim - 1))
    bps = MODULACIONES[modulacion]["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)

    series_ber = []
    for codigo in ["ninguno", "convolucional", "turbo"]:
        f = lambda snr, codigo=codigo: cadena_tx_codif_rx(
            rng.integers(0, 2, N_BITS_MC, dtype=np.uint8), modulacion, codigo,
            params, snr, rng)["ber"]
        t0 = time.perf_counter()
        ber_prom, ic_inf, ic_sup = _curva_ber_mc(f, snr_valores, n_sim, t_critico)
        tiempo_s = time.perf_counter() - t0               # tiempo total de esta curva (mismo trabajo por código)
        series_ber.append({"codigo": codigo, "ber_promedio": ber_prom,
                           "ic_inferior": ic_inf, "ic_superior": ic_sup, "tiempo_s": tiempo_s})

    return jsonify({
        "snr_valores": snr_valores,
        "n_simulaciones": n_sim,
        "t_critico": t_critico,
        "n_bits_mc": N_BITS_MC,
        "modulacion": modulacion,
        "series_ber": series_ber,
    })


# Punto de entrada: arranca el servidor de desarrollo Flask (puerto 5000 por defecto,
# configurable con la variable de entorno PORT).
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
