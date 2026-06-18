"""
=====================================================================================
 SC-FDM — Lógica de transmisión y recepción (solo para LEER y entender)
=====================================================================================
 Este archivo NO ejecuta nada (no hay servidor, ni Monte Carlo, ni imágenes). Solo
 contiene las funciones de la cadena SC-FDM, ordenadas en el orden en que ocurren los
 pasos: primero TRANSMISIÓN, luego el CANAL y al final la RECEPCIÓN.

 SC-FDM = OFDM con una DFT extra. Lo que lo distingue de OFDM:
   · En el TX se aplica una DFT de tamaño M al bloque de datos ANTES de la IFFT
     (precodificación "DFT-spread") y el bloque se coloca en subportadoras CONTIGUAS
     (mapeo localizado). Esto le da la propiedad de portadora única → PAPR bajo.
   · En el RX se aplica la IDFT inversa después de ecualizar.

 Flujo de un símbolo SC-FDM:
   bits → QAM → DFT-spread → mapeo localizado → IFFT+CP → [canal+ruido]
        → FFT → ecualizar → IDFT-despread → QAM → bits
=====================================================================================
"""

import numpy as np


# =====================================================================================
# ||                                TRANSMISIÓN                                       ||
# =====================================================================================

# _____________________________________________________________________________________
#  MAPEO QPSK                                                                Transmisión
#  Entran bits (0/1) y salen símbolos QPSK del plano I/Q (energía media = 1).
# _____________________________________________________________________________________
def mapear_qpsk(bits):
    bits = bits.reshape(-1, 2).astype(np.int8)   # Agrupa los bits de 2 en 2 (2 bits/símbolo)
    i = 1 - 2 * bits[:, 0]                        # Componente en fase (I): bit 0 → +1, bit 1 → -1
    q = 1 - 2 * bits[:, 1]                        # Componente en cuadratura (Q)
    return (i + 1j * q) / np.sqrt(2)             # I + jQ normalizado a energía unitaria (/√2)


# Mapa Gray de 4-PAM (un eje de 16-QAM): 2 bits → nivel de amplitud
_GRAY_4 = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}

# _____________________________________________________________________________________
#  MAPEO 16-QAM                                                              Transmisión
#  Entran bits y salen símbolos 16-QAM (Gray, energía media = 1). 4 bits/símbolo.
# _____________________________________________________________________________________
def mapear_16qam(bits):
    bits = bits.reshape(-1, 4)                                           # 4 bits por símbolo
    i = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 0:2]])  # Eje I: primeros 2 bits
    q = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 2:4]])  # Eje Q: últimos 2 bits
    return (i + 1j * q) / np.sqrt(10)                                   # Normaliza por √10


# _____________________________________________________________________________________
#  DFT-SPREAD (precodificación, EL PASO CLAVE DE SC-FDM)                     Transmisión
#  Entra el bloque de M símbolos QAM y sale su DFT de tamaño M (normalizada /√M). Esta
#  DFT es lo que convierte OFDM en SC-FDM y le da el PAPR bajo de portadora única.
# _____________________________________________________________________________________
def dft_spread(bloque_datos):
    m = len(bloque_datos)                  # Tamaño M (= nº de símbolos de datos)
    if m == 0:
        return bloque_datos
    return np.fft.fft(bloque_datos) / np.sqrt(m)   # DFT unitaria (conserva E_s = 1)


# _____________________________________________________________________________________
#  MAPEO LOCALIZADO (bloque contiguo)                                       Transmisión
#  Entra el bloque ya pasado por la DFT-spread y sale la rejilla de n_sc subportadoras
#  con ese bloque colocado en posiciones CONTIGUAS al inicio; el resto queda como guarda.
#  (SC-FDM NO intercala pilotos: eso conservaría la propiedad de portadora única.)
# _____________________________________________________________________________________
def mapeo_localizado(bloque_freq, n_sc):
    rejilla = np.zeros(n_sc, dtype=complex)         # Rejilla vacía (todo guarda)
    rejilla[:len(bloque_freq)] = bloque_freq        # Bloque contiguo al inicio
    return rejilla


# _____________________________________________________________________________________
#  MAPEO SUBPORTADORAS → BINS DE LA FFT                                     Transmisión
#  Entra la rejilla de n_sc subportadoras y sale el vector de n_fft puntos en banda
#  base (DC vacía y subportadoras centradas en ±frecuencia) y los índices FFT usados.
# _____________________________________________________________________________________
def mapeo_sc_a_fft(rejilla_sc, n_fft):
    n_sc = len(rejilla_sc)
    mitad = n_sc // 2
    indices_pos = np.arange(1, mitad + 1)                       # Frecuencias positivas
    indices_neg = n_fft - np.arange(1, mitad + 1)[::-1]         # Frecuencias negativas
    indices_fft = np.concatenate([indices_neg, indices_pos])
    vec = np.zeros(n_fft, dtype=complex)
    vec[indices_fft] = rejilla_sc
    return vec, indices_fft


# _____________________________________________________________________________________
#  MODULACIÓN OFDM (IFFT + PREFIJO CÍCLICO)                                  Transmisión
#  Entra la rejilla de subportadoras y sale la señal en el tiempo con el prefijo cíclico
#  (CP) antepuesto. (SC-FDM reusa la misma IFFT+CP de OFDM.)
# _____________________________________________________________________________________
def modulacion_ofdm(rejilla_sc, n_fft, n_cp):
    vec_freq, indices_fft = mapeo_sc_a_fft(rejilla_sc, n_fft)
    senal_t = np.fft.ifft(vec_freq) * n_fft / np.sqrt(len(rejilla_sc))   # IFFT (normalizada)
    cp = senal_t[-n_cp:] if n_cp > 0 else np.array([], dtype=complex)   # CP = últimas n_cp muestras
    return np.concatenate([cp, senal_t]), indices_fft                  # Antepone el CP


# =====================================================================================
# ||                                   CANAL                                          ||
# =====================================================================================

# Perfil potencia-retardo del canal Pedestrian A (ITU-R M.1225): 4 trayectorias (taps)
_RETARDOS_PEDA_NS = np.array([0.0, 110.0, 190.0, 410.0])     # Retardo de cada tap (ns)
_POTENCIAS_PEDA_DB = np.array([0.0, -9.7, -19.2, -22.8])     # Potencia relativa de cada tap (dB)

# _____________________________________________________________________________________
#  CANAL PEDESTRIAN A                                                              Canal
#  No entra señal: genera la respuesta en frecuencia H[k] del canal multitrayecto
#  (desvanecimiento de Rayleigh por tap) sobre las subportadoras activas.
# _____________________________________________________________________________________
def generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng):
    pot_lin = 10 ** (_POTENCIAS_PEDA_DB / 10)            # Potencias de los taps de dB a lineal
    pot_lin = pot_lin / np.sum(pot_lin)                  # Normaliza la potencia total a 1
    a = (rng.standard_normal(len(pot_lin)) + 1j * rng.standard_normal(len(pot_lin))) \
        * np.sqrt(pot_lin / 2)                           # Coeficiente complejo gaussiano por tap
    retardos_muestras = np.round(_RETARDOS_PEDA_NS * 1e-9 * fs).astype(int)  # Retardos → muestras
    h = np.zeros(n_fft, dtype=complex)
    for tap, m in enumerate(retardos_muestras):
        if m < n_fft:
            h[m] += a[tap]
    H = np.fft.fft(h)                                    # Respuesta en frecuencia = FFT de h
    return H[indices_fft]


# _____________________________________________________________________________________
#  RUIDO AWGN                                                                      Canal
#  Entra una señal y sale la misma señal con ruido blanco gaussiano complejo, calibrado
#  a la relación señal-ruido (SNR) indicada en dB.
# _____________________________________________________________________________________
def agregar_ruido_awgn(senal, snr_db, rng):
    pot_senal = np.mean(np.abs(senal) ** 2)
    pot_ruido = pot_senal / (10 ** (snr_db / 10))
    ruido = np.sqrt(pot_ruido / 2) * (rng.standard_normal(senal.shape) +
                                       1j * rng.standard_normal(senal.shape))
    return senal + ruido


# =====================================================================================
# ||                                  RECEPCIÓN                                       ||
# =====================================================================================

# _____________________________________________________________________________________
#  DEMODULACIÓN OFDM (QUITAR CP + FFT)                                        Recepción
#  Entra la señal recibida en el tiempo y salen las n_sc subportadoras (en frecuencia):
#  se descarta el prefijo cíclico y se aplica la FFT.
# _____________________________________________________________________________________
def demodulacion_ofdm(simbolo_con_cp, indices_fft, n_fft, n_cp, n_sc):
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]                  # Descarta el prefijo cíclico
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)       # FFT (normalización inversa)
    return espectro[indices_fft]


# _____________________________________________________________________________________
#  ECUALIZACIÓN ZERO-FORCING                                                  Recepción
#  Entran las subportadoras recibidas Y[k] y el canal H[k]; salen corregidas dividiendo
#  por el canal (X_est = Y / H).
# _____________________________________________________________________________________
def ecualizar_zf(Y, H):
    return Y / H


# _____________________________________________________________________________________
#  EXTRAER BLOQUE LOCALIZADO                                                  Recepción
#  Entra la rejilla ecualizada y salen las primeras M subportadoras (el bloque contiguo
#  donde viajaban los datos en SC-FDM).
# _____________________________________________________________________________________
def extraer_bloque_localizado(rejilla, m):
    return rejilla[:m]


# _____________________________________________________________________________________
#  IDFT-DESPREAD (deshace la precodificación, EL PASO CLAVE DE SC-FDM)        Recepción
#  Entra el bloque de M subportadoras ya ecualizado y salen otra vez los símbolos QAM:
#  es la IDFT inversa exacta de la DFT-spread del transmisor (·√M deshace el /√M).
# _____________________________________________________________________________________
def idft_despread(bloque_freq):
    m = len(bloque_freq)
    if m == 0:
        return bloque_freq
    return np.fft.ifft(bloque_freq) * np.sqrt(m)


# _____________________________________________________________________________________
#  DEMAPEO QPSK                                                               Recepción
#  Entran símbolos QPSK ruidosos y salen los bits decididos por signo (decisión dura).
# _____________________________________________________________________________________
def demapear_qpsk(simbolos):
    s = simbolos * np.sqrt(2)                     # Deshace la normalización (~{±1})
    b0 = (np.real(s) < 0).astype(np.uint8)       # Bit 0 = signo de la parte real
    b1 = (np.imag(s) < 0).astype(np.uint8)       # Bit 1 = signo de la parte imaginaria
    out = np.empty(2 * len(simbolos), dtype=np.uint8)
    out[0::2] = b0
    out[1::2] = b1
    return out


_GRAY_4_INV = {v: k for k, v in _GRAY_4.items()}   # Nivel → par de bits (16-QAM)

# _____________________________________________________________________________________
#  DEMAPEO 16-QAM                                                             Recepción
#  Entran símbolos 16-QAM ruidosos y salen los bits, decidiendo el nivel 4-PAM más
#  cercano en cada eje (I y Q).
# _____________________________________________________________________________________
def demapear_16qam(simbolos):
    s = simbolos * np.sqrt(10)
    niveles = [-3, -1, 1, 3]
    bi = _decidir_pam(np.real(s), niveles, _GRAY_4_INV, 2)   # Eje I → 2 bits
    bq = _decidir_pam(np.imag(s), niveles, _GRAY_4_INV, 2)   # Eje Q → 2 bits
    out = np.empty(4 * len(simbolos), dtype=np.uint8)
    out[0::4], out[1::4] = bi[:, 0], bi[:, 1]
    out[2::4], out[3::4] = bq[:, 0], bq[:, 1]
    return out


# _____________________________________________________________________________________
#  DECISIÓN DURA EN UN EJE PAM (apoyo del demapeo)                            Recepción
#  Entra un eje real, sus niveles válidos y el mapa nivel→bits; sale la matriz de bits
#  eligiendo el nivel más cercano a cada muestra.
# _____________________________________________________________________________________
def _decidir_pam(x, niveles, mapa_inverso, n_bits):
    niveles = np.array(niveles)
    idx = np.argmin(np.abs(x[:, None] - niveles[None, :]), axis=1)
    bits = np.empty((len(x), n_bits), dtype=np.uint8)
    for k, n in enumerate(niveles[idx]):
        bits[k, :] = mapa_inverso[int(n)]
    return bits


# =====================================================================================
# ||           FLUJO COMPLETO — cómo se encadenan los pasos (un símbolo SC-FDM)       ||
# =====================================================================================

# _____________________________________________________________________________________
#  CADENA SC-FDM (TX → CANAL → RX)                                        Todo el flujo
#  Entran M símbolos QAM de datos y salen los M símbolos recuperados. La diferencia con
#  OFDM son las dos cajas marcadas: DFT-spread al inicio e IDFT-despread al final.
# _____________________________________________________________________________________
def cadena_scfdm(simbolos_datos, n_sc, n_fft, n_cp, fs, snr_db, rng):
    m = len(simbolos_datos)

    # --- TRANSMISIÓN ---
    bloque_freq = dft_spread(simbolos_datos)        # ← PASO CLAVE: DFT-spread de M datos
    X = mapeo_localizado(bloque_freq, n_sc)         # Bloque contiguo en la rejilla
    _, indices_fft = mapeo_sc_a_fft(X, n_fft)       # Qué bins de la FFT se usan

    # --- CANAL: multiplica cada subportadora por H[k]; el ruido se suma en el tiempo ---
    H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)
    senal_canal, _ = modulacion_ofdm(X * H, n_fft, n_cp)     # IFFT+CP de X·H
    senal_rx = agregar_ruido_awgn(senal_canal, snr_db, rng)  # AWGN

    # --- RECEPCIÓN ---
    Y = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)   # Quita CP + FFT
    X_est = ecualizar_zf(Y, H)                                        # Divide por H[k]
    bloque_rx = extraer_bloque_localizado(X_est, m)                  # Toma el bloque contiguo
    return idft_despread(bloque_rx)                                  # ← PASO CLAVE: IDFT-despread
