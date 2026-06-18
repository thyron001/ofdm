"""
==========================================================================================
 PRÁCTICA 1 — OFDM (Orthogonal Frequency Division Multiplexing)
==========================================================================================
 Archivo AUTOCONTENIDO: no importa nada de app.py ni de los otros archivos de práctica.
 Todo lo que necesita OFDM está aquí, aunque se repita en otros archivos (es a propósito,
 para poder explicar esta práctica leyendo un solo archivo).

 ORDEN DEL ARCHIVO (de arriba hacia abajo):
   1) LÓGICA DE LA PRÁCTICA  -> lo importante a explicar: cómo se arma y recupera un
                                símbolo OFDM (pilotos, IFFT/FFT + CP, canal, ecualización).
   2) SOPORTE                -> modulaciones QAM, parámetros LTE, canal, ruido, BER, imagen.
   3) MONTE CARLO + FLASK    -> generación de curvas y endpoints web (no es el foco).

 Idea de OFDM (Práctica 1): los bits se modulan en símbolos QAM, se reparten en muchas
 subportadoras ortogonales, se intercalan pilotos conocidos para estimar el canal, se pasa
 al tiempo con una IFFT y se antepone un prefijo cíclico (CP). En el receptor se hace lo
 inverso y se ecualiza dividiendo por la respuesta del canal H[k] (Zero-Forcing).
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
# ===                       1) LÓGICA DE LA PRÁCTICA — OFDM                          ===
# =====================================================================================
# Esta es la parte central que se explica en la práctica. Las funciones de aquí construyen
# un símbolo OFDM y lo recuperan tras pasar por el canal.

PASO_PILOTO = 12  # Se inserta 1 subportadora piloto (valor conocido) cada 12 subportadoras


def indices_pilotos_datos(n_sc: int, paso: int = PASO_PILOTO) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve los índices (en [0, n_sc-1]) de las subportadoras piloto y de datos."""
    indices = np.arange(n_sc)                                     # Todas las subportadoras
    pilotos = indices[::paso]                                     # Pilotos: una de cada `paso` (12)
    datos = np.setdiff1d(indices, pilotos, assume_unique=False)   # El resto son datos
    return pilotos, datos


def insertar_pilotos(simbolos_datos: np.ndarray, n_sc: int,
                     paso: int = PASO_PILOTO, valor_piloto: complex = 1 + 0j) -> np.ndarray:
    """Arma la rejilla OFDM de n_sc subportadoras: datos + pilotos intercalados cada `paso`."""
    pilotos, datos = indices_pilotos_datos(n_sc, paso)   # Posiciones de pilotos y de datos
    rejilla = np.zeros(n_sc, dtype=complex)              # Rejilla vacía
    rejilla[pilotos] = valor_piloto                      # Coloca el valor piloto conocido (1+0j)
    n = min(len(datos), len(simbolos_datos))             # Cuántos símbolos de datos caben
    rejilla[datos[:n]] = simbolos_datos[:n]              # Coloca los símbolos de datos en su sitio
    return rejilla


def mapeo_sc_a_fft(rejilla_sc: np.ndarray, n_fft: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mapea las N_SC subportadoras al vector de N_FFT puntos en banda base, dejando la
    componente DC vacía y centrando los índices en [-N_SC/2,...,-1,+1,...,+N_SC/2].
    Retorna (vector_frecuencia, índices_fft_usados).
    """
    n_sc = len(rejilla_sc)                                       # Número de subportadoras activas
    mitad = n_sc // 2                                            # Mitad para repartir en frecuencias ±
    indices_pos = np.arange(1, mitad + 1)                       # Índices de frecuencias positivas
    indices_neg = n_fft - np.arange(1, mitad + 1)[::-1]         # Índices de frecuencias negativas
    indices_fft = np.concatenate([indices_neg, indices_pos])    # Orden monótono en frecuencia
    vec = np.zeros(n_fft, dtype=complex)                        # Vector espectral lleno de ceros (DC/guardas)
    vec[indices_fft] = rejilla_sc                               # Coloca las subportadoras activas
    return vec, indices_fft


def modulacion_ofdm(rejilla_sc: np.ndarray, n_fft: int, n_cp: int) -> Tuple[np.ndarray, np.ndarray]:
    """IFFT + prefijo cíclico (CP): convierte la rejilla de subportadoras en señal de tiempo."""
    vec_freq, indices_fft = mapeo_sc_a_fft(rejilla_sc, n_fft)            # Vector espectral centrado
    senal_t = np.fft.ifft(vec_freq) * n_fft / np.sqrt(len(rejilla_sc))  # IFFT con normalización de energía
    cp = senal_t[-n_cp:] if n_cp > 0 else np.array([], dtype=complex)   # CP = últimas n_cp muestras
    return np.concatenate([cp, senal_t]), indices_fft                  # Antepone el CP a la señal


def demodulacion_ofdm(simbolo_con_cp: np.ndarray, indices_fft: np.ndarray,
                       n_fft: int, n_cp: int, n_sc: int) -> np.ndarray:
    """Quita el CP, aplica la FFT y extrae las N_SC subportadoras activas."""
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]                  # Descarta el prefijo cíclico
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)       # FFT con normalización inversa
    return espectro[indices_fft]                               # Devuelve solo las subportadoras activas


def cadena_tx_rx(bits_tx: np.ndarray, modulacion: str, params: Dict,
                  snr_db: float, rng: np.random.Generator,
                  capturar_constelaciones: bool = False) -> Dict:
    """
    NÚCLEO DE LA PRÁCTICA 1. Ejecuta la cadena completa OFDM para un vector de bits:

        bits -> QAM -> [datos + pilotos] -> IFFT+CP -> canal Pedestrian A -> AWGN
             -> quitar CP + FFT -> ecualización Zero-Forcing (Y/H) -> demapeo -> bits

    Devuelve un dict con los bits recibidos, la BER y, opcionalmente, las constelaciones.
    """
    mod = MODULACIONES[modulacion]
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    n_datos = params["n_datos"]
    fs = params["fs"]

    # --- TX: bits -> símbolos QAM (con relleno a múltiplos exactos) ---
    bps = mod["bits"]                                  # Bits por símbolo de la modulación
    falta = (-len(bits_tx)) % bps                      # Bits que faltan para múltiplo de bps
    if falta:                                          # Rellena con ceros si no es múltiplo
        bits_tx_pad = np.concatenate([bits_tx, np.zeros(falta, dtype=np.uint8)])
    else:
        bits_tx_pad = bits_tx
    simbolos_qam = mod["mapear"](bits_tx_pad)         # Mapea todos los bits a símbolos QAM

    # Rellena hasta completar un número entero de símbolos OFDM
    falta_sim = (-len(simbolos_qam)) % n_datos
    if falta_sim:
        simbolos_qam = np.concatenate([simbolos_qam, np.zeros(falta_sim, dtype=complex)])
    n_ofdm = len(simbolos_qam) // n_datos             # Nº de símbolos OFDM a transmitir

    constelacion_tx = simbolos_qam.copy() if capturar_constelaciones else None
    simbolos_rx_datos = []                            # Acumulador de símbolos RX (constelación)
    bits_rx_total = []                                # Acumulador de bits recibidos
    _, indices_datos_sc = indices_pilotos_datos(n_sc)  # Posiciones de las subportadoras de datos

    for i in range(n_ofdm):                            # Procesa símbolo a símbolo
        bloque_datos = simbolos_qam[i * n_datos:(i + 1) * n_datos]   # Bloque de n_datos símbolos
        rejilla = insertar_pilotos(bloque_datos, n_sc)              # TX: datos + pilotos intercalados
        _, indices_fft = modulacion_ofdm(rejilla, n_fft, n_cp)      # IFFT + CP (aquí solo para indices_fft)

        # Canal Pedestrian A + ecualización Zero-Forcing (1 antena)
        H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)  # Respuesta del canal H[k]
        senal_canal, _ = modulacion_ofdm(rejilla * H, n_fft, n_cp)   # Y[k]=H[k]·X[k], vuelve al tiempo
        senal_rx = agregar_ruido_awgn(senal_canal, snr_db, rng)      # AWGN al SNR objetivo
        rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)  # Quita CP + FFT
        rejilla_eq = rejilla_rx / H                                  # Ecualización ZF: divide por H[k]

        datos_rx = rejilla_eq[indices_datos_sc]                      # Toma las posiciones de datos
        if capturar_constelaciones:
            simbolos_rx_datos.append(datos_rx)
        bits_rx_total.append(mod["demapear"](datos_rx))             # Demapea a bits y acumula

    bits_rx = np.concatenate(bits_rx_total) if bits_rx_total else np.array([], dtype=np.uint8)
    bits_rx = bits_rx[:len(bits_tx_pad)]              # Recorta al tamaño transmitido
    ber = calcular_ber(bits_tx_pad, bits_rx)          # Calcula la BER

    salida = {
        "bits_rx": bits_rx[:len(bits_tx)],            # Bits útiles (sin el padding de bits)
        "ber": ber,
        "n_simbolos_ofdm": n_ofdm,
    }
    if capturar_constelaciones:
        salida["constelacion_tx"] = constelacion_tx
        salida["constelacion_rx"] = np.concatenate(simbolos_rx_datos)
    return salida


# =====================================================================================
# ===          2) SOPORTE — MODULACIONES QAM, PARÁMETROS, CANAL, RUIDO, IMAGEN       ===
# =====================================================================================
# Esto es maquinaria de apoyo: no es el foco de la práctica, pero la cadena de arriba la usa.

# --- Modulaciones digitales QAM (Gray, energía media unitaria) ---

def qpsk_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a símbolos QPSK normalizados (energía media unitaria)."""
    bits = bits.reshape(-1, 2).astype(np.int8)   # Agrupa los bits de dos en dos (2 bits/símbolo)
    i = 1 - 2 * bits[:, 0]                        # Componente en fase (I) a partir del primer bit
    q = 1 - 2 * bits[:, 1]                        # Componente en cuadratura (Q) a partir del segundo bit
    return (i + 1j * q) / np.sqrt(2)             # Combina I+jQ y normaliza a energía unitaria (/√2)


def qpsk_demapear(simbolos: np.ndarray) -> np.ndarray:
    """Demapeo por decisión dura (hard-decision) de QPSK."""
    s = simbolos * np.sqrt(2)                     # Deshace la normalización para volver a {±1}
    b0 = (np.real(s) < 0).astype(np.uint8)       # Bit 0 = signo de la parte real (negativa -> 1)
    b1 = (np.imag(s) < 0).astype(np.uint8)       # Bit 1 = signo de la parte imaginaria
    out = np.empty(2 * len(simbolos), dtype=np.uint8)
    out[0::2] = b0
    out[1::2] = b1
    return out


_GRAY_4 = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}   # Mapa Gray 4-PAM (eje de 16-QAM)
_GRAY_4_INV = {v: k for k, v in _GRAY_4.items()}


def qam16_mapear(bits: np.ndarray) -> np.ndarray:
    """Mapea bits a 16-QAM con codificación Gray, normalizado (E_s = 1)."""
    bits = bits.reshape(-1, 4)                                            # 4 bits por símbolo
    i = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 0:2]])   # Eje I: primeros 2 bits -> 4-PAM
    q = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 2:4]])   # Eje Q: últimos 2 bits -> 4-PAM
    return (i + 1j * q) / np.sqrt(10)                                    # Normaliza por √10


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


_GRAY_8 = {                                                    # Mapa Gray 8-PAM (eje de 64-QAM)
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


# Asocia cada modulación con sus bits/símbolo y sus funciones de mapeo
MODULACIONES = {
    "QPSK":   {"bits": 2, "mapear": qpsk_mapear,  "demapear": qpsk_demapear},
    "16-QAM": {"bits": 4, "mapear": qam16_mapear, "demapear": qam16_demapear},
    "64-QAM": {"bits": 6, "mapear": qam64_mapear, "demapear": qam64_demapear},
}

# OFDM admite las 3 modulaciones
MODS_OFDM = ["QPSK", "16-QAM", "64-QAM"]


# --- Parámetros de la numerología LTE ---

TABLA_NSC = {                                                  # (BW_MHz, Δf_kHz) -> N_SC
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

    n_sc = TABLA_NSC[clave]                               # Número de subportadoras útiles
    n_fft = siguiente_pot_2(n_sc)                         # Tamaño de FFT (potencia de 2 >= N_SC)
    fs = n_fft * delta_f_khz * 1e3                        # Frecuencia de muestreo fs = N_FFT·Δf (Hz)
    dur_cp = duracion_cp_us(delta_f_khz, tipo_cp) * 1e-6  # Duración del CP en segundos
    n_cp = int(round(dur_cp * fs))                        # Longitud del CP en muestras

    n_pilotos = n_sc // PASO_PILOTO                       # Subportadoras piloto (1 de cada 12)
    n_datos = n_sc - n_pilotos                            # Subportadoras de datos

    n_simbolos_qam = math.ceil(n_bits / bits_por_simbolo) if bits_por_simbolo > 0 else 0
    n_simbolos_ofdm = math.ceil(n_simbolos_qam / n_datos) if n_datos > 0 else 0
    duracion_simbolo_us = (n_fft + n_cp) / fs * 1e6      # Duración total del símbolo (FFT+CP) en µs

    return {
        "n_sc": int(n_sc), "n_fft": int(n_fft), "fs": float(fs), "n_cp": int(n_cp),
        "n_pilotos": int(n_pilotos), "n_datos": int(n_datos),
        "n_simbolos_qam": int(n_simbolos_qam), "n_simbolos_ofdm": int(n_simbolos_ofdm),
        "duracion_simbolo_us": float(duracion_simbolo_us), "duracion_cp_us": float(dur_cp * 1e6),
        "delta_f_hz": float(delta_f_khz * 1e3), "bw_hz": float(bw_mhz * 1e6),
    }


# --- Canal Pedestrian A, ruido AWGN, BER y PAPR ---

RETARDOS_PEDA_NS = np.array([0.0, 110.0, 190.0, 410.0])     # Retardos de los taps (ns)
POTENCIAS_PEDA_DB = np.array([0.0, -9.7, -19.2, -22.8])     # Potencia relativa de los taps (dB)


def generar_canal_pedestrian_a(n_fft: int, fs: float, indices_fft: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    """Respuesta en frecuencia H[k] del canal Pedestrian A (Rayleigh por tap)."""
    pot_lin = 10 ** (POTENCIAS_PEDA_DB / 10)             # Potencias de los taps de dB a lineal
    pot_lin = pot_lin / np.sum(pot_lin)                  # Normaliza la potencia total a 1
    a = (rng.standard_normal(len(pot_lin)) + 1j * rng.standard_normal(len(pot_lin))) \
        * np.sqrt(pot_lin / 2)                           # Coeficiente complejo gaussiano por tap
    retardos_muestras = np.round(RETARDOS_PEDA_NS * 1e-9 * fs).astype(int)  # Retardos -> muestras
    h = np.zeros(n_fft, dtype=complex)                   # Respuesta al impulso discreta
    for tap, m in enumerate(retardos_muestras):
        if m < n_fft:
            h[m] += a[tap]
    H = np.fft.fft(h)                                    # Respuesta en frecuencia = FFT de h
    return H[indices_fft]                               # Solo en las subportadoras activas


def agregar_ruido_awgn(senal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Agrega ruido blanco gaussiano complejo (AWGN) calibrado al SNR indicado (en dB)."""
    pot_senal = np.mean(np.abs(senal) ** 2)             # Potencia media de la señal
    pot_ruido = pot_senal / (10 ** (snr_db / 10))       # Potencia de ruido para el SNR objetivo
    ruido = np.sqrt(pot_ruido / 2) * (rng.standard_normal(senal.shape) +
                                       1j * rng.standard_normal(senal.shape))
    return senal + ruido


def calcular_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    """Tasa de error de bit (BER) = bits distintos / bits comparados."""
    n = min(len(bits_tx), len(bits_rx))
    if n == 0:
        return 0.0
    return float(np.sum(bits_tx[:n] != bits_rx[:n]) / n)


def calcular_papr_simbolo(senal_tiempo: np.ndarray) -> float:
    """PAPR (en dB) de un símbolo en el dominio del tiempo (sin CP)."""
    pot_pico = np.max(np.abs(senal_tiempo) ** 2)        # Potencia instantánea máxima (pico)
    pot_media = np.mean(np.abs(senal_tiempo) ** 2)      # Potencia media
    if pot_media <= 0:
        return 0.0
    return float(10 * np.log10(pot_pico / pot_media))


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
# Generación de la forma de onda, la CCDF del PAPR y las curvas BER vs SNR por Monte Carlo,
# además del servidor web que expone todo a la interfaz. No es lo central de la práctica.

def forma_onda_simbolo(simbolos_qam_bloque: np.ndarray, params: Dict,
                        max_pts: int = 512) -> Tuple[list, float]:
    """Genera UN símbolo OFDM y devuelve (envolvente |x(t)| submuestreada, PAPR en dB)."""
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    n_datos = params["n_datos"]

    bloque = np.asarray(simbolos_qam_bloque, dtype=complex)[:n_datos]
    if len(bloque) < n_datos:
        bloque = np.concatenate([bloque, np.zeros(n_datos - len(bloque), dtype=complex)])
    rejilla = insertar_pilotos(bloque, n_sc)                 # Mapeo OFDM (datos + pilotos)
    senal_t, _ = modulacion_ofdm(rejilla, n_fft, n_cp)       # IFFT + CP
    senal = senal_t[n_cp:]                                   # Símbolo puro, sin CP, para el PAPR
    papr = calcular_papr_simbolo(senal)
    envolvente = np.abs(senal)
    if len(envolvente) > max_pts:
        idx = np.linspace(0, len(envolvente) - 1, max_pts).astype(int)
        envolvente = envolvente[idx]
    return envolvente.tolist(), float(papr)


def calcular_papr_ccdf(modulacion: str, params: Dict, n_simbolos: int,
                        rng: np.random.Generator, eje_x: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estima la CCDF del PAPR de OFDM generando n_simbolos aleatorios. ccdf[i]=Pr{PAPR>x[i]}."""
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
        rejilla = insertar_pilotos(simbolos_qam, n_sc)
        senal_t, _ = modulacion_ofdm(rejilla, n_fft, n_cp)
        paprs[i] = calcular_papr_simbolo(senal_t[n_cp:])

    if eje_x is None:
        eje_x = np.linspace(0, max(15.0, float(np.max(paprs)) + 0.5), 80)
    ccdf = np.array([np.mean(paprs > x) for x in eje_x])
    return eje_x, ccdf


# --- Servidor Flask ---

app = Flask(__name__)
CARPETA_UPLOADS = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(CARPETA_UPLOADS, exist_ok=True)
app.config["UPLOAD_FOLDER"] = CARPETA_UPLOADS

# Estado en memoria de la última imagen subida (aplicación mono-usuario, de demostración)
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


@app.route("/simular", methods=["POST"])
def simular():
    """Ejecuta UNA transmisión OFDM de la imagen cargada y devuelve imagen, BER, constelaciones."""
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
        snr = float(data["snr_db"])
        modulacion = data["modulacion"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    if modulacion not in MODS_OFDM:
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400
    if ESTADO["bits"] is None:
        return jsonify({"error": "Primero suba una imagen"}), 400

    bps = MODULACIONES[modulacion]["bits"]
    bits_tx = ESTADO["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, len(bits_tx), bps)

    rng = np.random.default_rng()
    t0 = time.perf_counter()
    resultado = cadena_tx_rx(bits_tx, modulacion, params, snr, rng, capturar_constelaciones=True)
    tiempo_computo_s = time.perf_counter() - t0

    # Forma de onda |x(t)| de un símbolo OFDM con datos aleatorios (la imagen daría un PAPR atípico)
    n_bits_bloque = params["n_datos"] * bps
    bits_bloque = rng.integers(0, 2, n_bits_bloque, dtype=np.uint8)
    simbolos_bloque = MODULACIONES[modulacion]["mapear"](bits_bloque)
    env_ofdm, papr_ofdm = forma_onda_simbolo(simbolos_bloque, params)

    tiempo_aire_s = resultado["n_simbolos_ofdm"] * params["duracion_simbolo_us"] * 1e-6
    img_rx = bits_a_imagen(resultado["bits_rx"], ESTADO["imagen_mode"], ESTADO["imagen_size"])
    img_tx = Image.open(io.BytesIO(ESTADO["imagen_bytes"]))

    def submuestrear(arr, max_n=3000):
        if len(arr) > max_n:
            idx = np.random.choice(len(arr), max_n, replace=False)
            return arr[idx]
        return arr

    c_tx = submuestrear(resultado["constelacion_tx"])
    c_rx = submuestrear(resultado["constelacion_rx"])
    pilotos_idx, datos_idx = indices_pilotos_datos(params["n_sc"])

    return jsonify({
        "ber": resultado["ber"],
        "esquema": "OFDM",
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
            "guarda": 0,
            "tipo_mapeo": "distribuido",
            "indices_pilotos": pilotos_idx.tolist(),
            "indices_datos": datos_idx.tolist(),
            "indices_guarda": [],
        },
        "forma_onda": {"ofdm": {"envolvente": env_ofdm, "papr_db": papr_ofdm}},
        "parametros": params,
        "tiempo_aire_s": tiempo_aire_s,
        "tiempo_computo_s": tiempo_computo_s,
        "throughput_bps": (len(bits_tx) / tiempo_aire_s) if tiempo_aire_s > 0 else 0.0,
    })


@app.route("/montecarlo", methods=["POST"])
def montecarlo():
    """Monte Carlo OFDM: BER vs SNR (IC 95%) y CCDF del PAPR para las 3 modulaciones."""
    data = request.get_json(force=True)
    try:
        bw = float(data["bw_mhz"])
        df = float(data["delta_f_khz"])
        cp = data["tipo_cp"]
    except KeyError as e:
        return jsonify({"error": f"Falta campo {e}"}), 400

    N_BITS_MC = 50000                                  # Bits por corrida
    N_SIMBOLOS_PAPR = 1500                             # Símbolos para la CCDF del PAPR
    snr_valores = list(range(0, 16))                   # Barrido de SNR de 0 a 15 dB
    n_sim = 10                                         # Corridas independientes por punto
    rng = np.random.default_rng()
    t_critico = float(t_student.ppf(0.975, df=n_sim - 1))  # t de Student al 95%

    # --- BER vs SNR: una serie por modulación ---
    series_ber = []
    for modulacion in MODS_OFDM:
        bps = MODULACIONES[modulacion]["bits"]
        params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)
        ber_prom, ic_inf, ic_sup = [], [], []
        for snr in snr_valores:
            bers = []
            for _ in range(n_sim):
                bits_aleatorios = rng.integers(0, 2, N_BITS_MC, dtype=np.uint8)
                r = cadena_tx_rx(bits_aleatorios, modulacion, params, snr, rng)
                bers.append(r["ber"])
            bers = np.array(bers)
            mu = float(np.mean(bers))
            sd = float(np.std(bers, ddof=1)) if n_sim > 1 else 0.0
            margen = t_critico * sd / np.sqrt(n_sim)
            ber_prom.append(mu)
            ic_inf.append(max(mu - margen, 1e-6))
            ic_sup.append(mu + margen)
        series_ber.append({
            "esquema": "OFDM", "modulacion": modulacion,
            "ber_promedio": ber_prom, "ic_inferior": ic_inf, "ic_superior": ic_sup,
        })

    # --- CCDF del PAPR (independiente del SNR), eje común ---
    eje_comun = np.linspace(0, 15, 80)
    series_papr = []
    for modulacion in MODS_OFDM:
        bps = MODULACIONES[modulacion]["bits"]
        params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)
        _, c = calcular_papr_ccdf(modulacion, params, N_SIMBOLOS_PAPR, rng, eje_x=eje_comun)
        series_papr.append({"esquema": "OFDM", "modulacion": modulacion, "ccdf": c.tolist()})

    return jsonify({
        "snr_valores": snr_valores,
        "n_simulaciones": n_sim,
        "t_critico": t_critico,
        "n_bits_mc": N_BITS_MC,
        "series_ber": series_ber,
        "papr": {"x_db": eje_comun.tolist(), "series": series_papr, "n_simbolos": N_SIMBOLOS_PAPR},
    })


# Punto de entrada: arranca el servidor de desarrollo Flask en el puerto 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
