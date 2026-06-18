"""
==========================================================================================
 PRÁCTICA 4 — DIVERSIDAD EN TRANSMISIÓN (SFBC / código de Alamouti espacio-frecuencia)
==========================================================================================
 Archivo AUTOCONTENIDO: no importa nada de app.py ni de los otros archivos de práctica.
 El código repetido respecto a los demás archivos es a propósito, para poder explicar esta
 práctica leyendo un solo archivo.

 ORDEN DEL ARCHIVO (de arriba hacia abajo):
   1) LÓGICA DE LA PRÁCTICA  -> lo importante a explicar: el codificado de Alamouti por PARES
                                de subportadoras (2 antenas TX) y su combinador ortogonal.
   2) SOPORTE                -> modulaciones QAM, parámetros LTE, canal, ruido, BER, imagen.
   3) MONTE CARLO + FLASK    -> generación de curvas y endpoints web (no es el foco).

 Idea de la diversidad en TX (Práctica 4): con 2 antenas transmisoras se aplica el código de
 Alamouti en el dominio espacio-frecuencia (SFBC). Cada PAR de subportadoras adyacentes
 (k, k+1) lleva (s0, s1) por la antena 1 y (-s1*, s0*) por la antena 2, repartiendo la
 potencia 1/√2 por antena. Como las subportadoras del par son adyacentes, H[k] ≈ H[k+1] y el
 combinador ortogonal de Alamouti separa s0 y s1 sin interferencia, dando ganancia de
 diversidad (2x1 -> orden 2, 2x2 -> orden 4) sin necesidad de conocer el canal en el TX.
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
# ===             1) LÓGICA DE LA PRÁCTICA — DIVERSIDAD EN TX (SFBC)                 ===
# =====================================================================================
# Parte central de la práctica. El núcleo es cadena_tx_sfbc_rx (codificado de Alamouti por
# pares + combinador). Las funciones OFDM (IFFT/FFT + CP) son el soporte de la forma de onda
# que SFBC necesita; a diferencia de OFDM, SFBC NO usa pilotos: ocupa todas las subportadoras
# en pares (el CSI es "genie" en este simulador).


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


def cadena_tx_sfbc_rx(bits_tx: np.ndarray, modulacion: str, params: Dict,
                       snr_db: float, rng: np.random.Generator,
                       capturar_constelaciones: bool = False,
                       n_tx: int = 2, n_rx: int = 1) -> Dict:
    """
    NÚCLEO DE LA PRÁCTICA 4 (diversidad en TX con SFBC). El transmisor usa n_tx antenas y un
    mapeo por PARES de subportadoras adyacentes:

      - n_tx == 1 -> SISO de referencia (sin diversidad): canal Pedestrian A + ecualización ZF
                     (Y/H). Si n_rx > 1 se combinan las antenas RX con MRC. Diversidad = n_rx.
      - n_tx == 2 -> SFBC (Alamouti espacio-frecuencia): cada par (k, k+1) lleva (s0, s1) y, por
                     la 2ª antena, (-s1*, s0*). Cada antena se escala por 1/√2 para REPARTIR la
                     potencia total (penalización de ~3 dB vs MRC). Cada antena RX ve dos canales
                     independientes (H1, H2) y se decodifica con el combinador ortogonal:
                        s0_est = Σ_rx (conj(h1)·r0 + h2·conj(r1)) / Σ_rx (|h1|²+|h2|²)
                        s1_est = Σ_rx (conj(h1)·r1 - h2·conj(r0)) / Σ_rx (|h1|²+|h2|²)
                     Diversidad = 2·n_rx (2x1 -> 2, 2x2 -> 4).

    A diferencia de OFDM, SFBC NO intercala pilotos: usa TODAS las subportadoras en pares, así
    los dos elementos del par son adyacentes y se cumple H[k] ≈ H[k+1] (el canal Pedestrian A es
    muy suave en frecuencia). El único par que cruza el hueco de DC tiene un desajuste mínimo.
    """
    mod = MODULACIONES[modulacion]
    n_sc = params["n_sc"]
    n_fft = params["n_fft"]
    n_cp = params["n_cp"]
    fs = params["fs"]
    bps = mod["bits"]

    n_par = n_sc // 2                                  # Nº de pares Alamouti por símbolo OFDM
    n_datos = n_par * 2                                # Subportadoras de datos usadas (siempre par)

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

    # Índices FFT de las n_sc subportadoras activas (no dependen de los datos: rejilla vacía)
    _, indices_fft = modulacion_ofdm(np.zeros(n_sc, dtype=complex), n_fft, n_cp)
    esc = 1.0 / np.sqrt(2.0)                           # Reparto de potencia entre las 2 antenas TX

    for i in range(n_ofdm):
        bloque = simbolos_qam[i * n_datos:(i + 1) * n_datos]
        s0 = bloque[0::2]                              # Símbolo "par" de cada pareja (subportadora k)
        s1 = bloque[1::2]                              # Símbolo "impar" de cada pareja (subportadora k+1)

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
                senal_rx = agregar_ruido_awgn(senal_canal, snr_db, rng)
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
                senal_rx = agregar_ruido_awgn(senal1 + senal2, snr_db, rng)  # Se suman + AWGN
                rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
                r0 = rejilla_rx[0::2][:n_par]          # Recibido en la subportadora k
                r1 = rejilla_rx[1::2][:n_par]          # Recibido en la subportadora k+1
                h1 = H1[0::2][:n_par]                  # Canal TX1 en el par (H1[k] ≈ H1[k+1])
                h2 = H2[0::2][:n_par]                  # Canal TX2 en el par
                num0 += np.conj(h1) * r0 + h2 * np.conj(r1)   # Combinador Alamouti para s0
                num1 += np.conj(h1) * r1 - h2 * np.conj(r0)   # Combinador Alamouti para s1
                den += np.abs(h1) ** 2 + np.abs(h2) ** 2
            # /esc deshace el reparto 1/√2 y devuelve la constelación a escala unitaria
            # (imprescindible para que el demapeador de 16/64-QAM use las regiones correctas).
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

PASO_PILOTO = 12  # Solo se usa para los campos n_pilotos/n_datos de los parámetros (SFBC no usa pilotos)

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
    """Traduce la configuración (BW, Δf, CP) en los parámetros físicos de la cadena."""
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
# Curvas BER vs SNR por configuración (Monte Carlo) y el servidor web. No es lo central.

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


@app.route("/simular_sfbc", methods=["POST"])
def simular_sfbc():
    """
    Ejecuta UNA transmisión de la imagen con SFBC (Alamouti) en la configuración elegida
    (2x1 o 2x2). Para la comparación "antes vs después" ejecuta además una cadena SISO (1x1,
    sin diversidad) sobre los mismos bits. Devuelve la imagen recuperada, la BER y las
    constelaciones RX sin diversidad (antes) y con SFBC (después).
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
    if modulacion not in MODS_OFDM:
        return jsonify({"error": f"{modulacion} no está disponible en OFDM"}), 400
    if config not in CONFIGS:
        return jsonify({"error": f"Configuración '{config}' no válida (use 2x1 o 2x2)"}), 400
    if ESTADO["bits"] is None:
        return jsonify({"error": "Primero suba una imagen"}), 400

    n_tx, n_rx = CONFIGS[config]
    bps = MODULACIONES[modulacion]["bits"]
    bits_tx = ESTADO["bits"]
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

    def submuestrear(arr, max_n=3000):
        if len(arr) > max_n:
            idx = np.random.choice(len(arr), max_n, replace=False)
            return arr[idx]
        return arr

    c_rx = submuestrear(resultado["constelacion_rx"])
    c_antes = submuestrear(ref["constelacion_rx"])
    salida = {
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
    Monte Carlo de BER vs SNR para diversidad en TX. Calcula una curva por cada configuración
    en `configs` (def. ["1x1", "2x1", "2x2"]). Al subir el orden de diversidad (1 -> 2 -> 4) la
    curva debe caer con mayor pendiente. IC 95% con la t de Student.
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

    # Configuraciones: nombre -> (n_tx, n_rx). Diversidad = n_tx·n_rx: 1, 2, 4.
    CONFIGS = {"1x1": (1, 1), "2x1": (2, 1), "2x2": (2, 2)}
    configs = [c for c in data.get("configs", ["1x1", "2x1", "2x2"]) if c in CONFIGS]
    if not configs:
        configs = ["1x1", "2x1", "2x2"]

    N_BITS_MC = 10000                                  # Bits por corrida
    snr_valores = list(range(0, 16))                   # Barrido de SNR de 0 a 15 dB
    n_sim = 8                                          # Corridas independientes por punto
    rng = np.random.default_rng()
    t_critico = float(t_student.ppf(0.975, df=n_sim - 1))

    bps = MODULACIONES[modulacion]["bits"]
    params = calcular_parametros_ofdm(bw, df, cp, N_BITS_MC, bps)

    series_ber = []
    for cfg in configs:                                # Una curva BER vs SNR por configuración
        n_tx, n_rx = CONFIGS[cfg]
        ber_prom, ic_inf, ic_sup = [], [], []
        for snr in snr_valores:
            bers = []
            for _ in range(n_sim):
                bits_aleatorios = rng.integers(0, 2, N_BITS_MC, dtype=np.uint8)
                r = cadena_tx_sfbc_rx(bits_aleatorios, modulacion, params, snr, rng,
                                       capturar_constelaciones=False, n_tx=n_tx, n_rx=n_rx)
                bers.append(r["ber"])
            bers = np.array(bers)
            mu = float(np.mean(bers))
            sd = float(np.std(bers, ddof=1)) if n_sim > 1 else 0.0
            margen = t_critico * sd / np.sqrt(n_sim)
            ber_prom.append(mu)
            ic_inf.append(max(mu - margen, 1e-6))
            ic_sup.append(mu + margen)
        series_ber.append({
            "config": cfg,
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
