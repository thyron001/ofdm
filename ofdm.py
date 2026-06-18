"""
=====================================================================================
 OFDM — Lógica de transmisión y recepción (solo para LEER y entender)
=====================================================================================
 Este archivo NO ejecuta nada (no hay servidor, ni Monte Carlo, ni imágenes). Solo
 contiene las funciones de la cadena OFDM, ordenadas en el MISMO orden en que ocurren
 los pasos: primero TRANSMISIÓN, luego el CANAL y al final la RECEPCIÓN.

 Cada función lleva arriba un título y una descripción corta de qué entra y qué sale.

 Flujo de un símbolo OFDM:
   bits → QAM → pilotos → IFFT+CP →  [canal + ruido]  → FFT → ecualizar → quitar pilotos → QAM → bits
   \________________ TRANSMISIÓN ________________/                 \____________ RECEPCIÓN ____________/
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


# Mapa Gray de 8-PAM (un eje de 64-QAM): 3 bits → nivel de amplitud
_GRAY_8 = {
    (0, 0, 0): -7, (0, 0, 1): -5, (0, 1, 1): -3, (0, 1, 0): -1,
    (1, 1, 0):  1, (1, 1, 1):  3, (1, 0, 1):  5, (1, 0, 0):  7,
}

# _____________________________________________________________________________________
#  MAPEO 64-QAM                                                              Transmisión
#  Entran bits y salen símbolos 64-QAM (Gray, energía media = 1). 6 bits/símbolo.
# _____________________________________________________________________________________
def mapear_64qam(bits):
    bits = bits.reshape(-1, 6)                                                      # 6 bits/símbolo
    i = np.array([_GRAY_8[(int(a), int(b), int(c))] for a, b, c in bits[:, 0:3]])   # Eje I: 3 bits
    q = np.array([_GRAY_8[(int(a), int(b), int(c))] for a, b, c in bits[:, 3:6]])   # Eje Q: 3 bits
    return (i + 1j * q) / np.sqrt(42)                                              # Normaliza por √42


# _____________________________________________________________________________________
#  POSICIONES DE PILOTOS Y DATOS                                             Transmisión
#  Entra el nº de subportadoras y salen los índices de pilotos y de datos.
#  (Se usa al insertar pilotos en el TX y al quitarlos en el RX.)
# _____________________________________________________________________________________
def indices_pilotos_datos(n_sc, paso=12):
    indices = np.arange(n_sc)                                     # Todas las subportadoras
    pilotos = indices[::paso]                                     # 1 piloto cada `paso` (12)
    datos = np.setdiff1d(indices, pilotos, assume_unique=False)   # El resto son datos
    return pilotos, datos


# _____________________________________________________________________________________
#  INSERTAR PILOTOS (armar la rejilla)                                      Transmisión
#  Entran los símbolos QAM y sale la rejilla de n_sc subportadoras con los pilotos
#  (valor conocido 1+0j) intercalados cada 12 posiciones.
# _____________________________________________________________________________________
def insertar_pilotos(simbolos_datos, n_sc, paso=12, valor_piloto=1 + 0j):
    pilotos, datos = indices_pilotos_datos(n_sc, paso)   # Posiciones de pilotos y de datos
    rejilla = np.zeros(n_sc, dtype=complex)              # Rejilla vacía
    rejilla[pilotos] = valor_piloto                      # Coloca los pilotos conocidos
    n = min(len(datos), len(simbolos_datos))             # Cuántos símbolos de datos caben
    rejilla[datos[:n]] = simbolos_datos[:n]              # Coloca los datos en su sitio
    return rejilla


# _____________________________________________________________________________________
#  MAPEO SUBPORTADORAS → BINS DE LA FFT                                     Transmisión
#  Entra la rejilla de n_sc subportadoras y sale el vector de n_fft puntos en banda
#  base (DC vacía y subportadoras centradas en ±frecuencia). También devuelve qué
#  índices de la FFT se usaron.
# _____________________________________________________________________________________
def mapeo_sc_a_fft(rejilla_sc, n_fft):
    n_sc = len(rejilla_sc)                                       # Subportadoras activas
    mitad = n_sc // 2                                            # Mitad para frecuencias ±
    indices_pos = np.arange(1, mitad + 1)                       # Frecuencias positivas (+1..+mitad)
    indices_neg = n_fft - np.arange(1, mitad + 1)[::-1]         # Frecuencias negativas
    indices_fft = np.concatenate([indices_neg, indices_pos])    # Orden monótono en frecuencia
    vec = np.zeros(n_fft, dtype=complex)                        # Vector espectral (ceros = DC/guardas)
    vec[indices_fft] = rejilla_sc                              # Coloca las subportadoras activas
    return vec, indices_fft


# _____________________________________________________________________________________
#  MODULACIÓN OFDM (IFFT + PREFIJO CÍCLICO)                                  Transmisión
#  Entra la rejilla de subportadoras y sale la señal OFDM en el tiempo, con el prefijo
#  cíclico (CP) antepuesto. También devuelve los índices de la FFT usados.
# _____________________________________________________________________________________
def modulacion_ofdm(rejilla_sc, n_fft, n_cp):
    vec_freq, indices_fft = mapeo_sc_a_fft(rejilla_sc, n_fft)            # Vector espectral centrado
    senal_t = np.fft.ifft(vec_freq) * n_fft / np.sqrt(len(rejilla_sc))  # IFFT (con normalización)
    cp = senal_t[-n_cp:] if n_cp > 0 else np.array([], dtype=complex)   # CP = últimas n_cp muestras
    return np.concatenate([cp, senal_t]), indices_fft                  # Antepone el CP a la señal


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
    h = np.zeros(n_fft, dtype=complex)                   # Respuesta al impulso discreta
    for tap, m in enumerate(retardos_muestras):          # Coloca cada tap en su retardo
        if m < n_fft:
            h[m] += a[tap]
    H = np.fft.fft(h)                                    # Respuesta en frecuencia = FFT de h
    return H[indices_fft]                               # Solo en las subportadoras activas


# _____________________________________________________________________________________
#  RUIDO AWGN                                                                      Canal
#  Entra una señal y sale la misma señal con ruido blanco gaussiano complejo, calibrado
#  a la relación señal-ruido (SNR) indicada en dB.
# _____________________________________________________________________________________
def agregar_ruido_awgn(senal, snr_db, rng):
    pot_senal = np.mean(np.abs(senal) ** 2)             # Potencia media de la señal
    pot_ruido = pot_senal / (10 ** (snr_db / 10))       # Potencia de ruido para ese SNR
    ruido = np.sqrt(pot_ruido / 2) * (rng.standard_normal(senal.shape) +
                                       1j * rng.standard_normal(senal.shape))
    return senal + ruido


# =====================================================================================
# ||                                  RECEPCIÓN                                       ||
# =====================================================================================

# _____________________________________________________________________________________
#  DEMODULACIÓN OFDM (QUITAR CP + FFT)                                        Recepción
#  Entra la señal OFDM recibida en el tiempo y salen las n_sc subportadoras (en
#  frecuencia): se descarta el prefijo cíclico y se aplica la FFT.
# _____________________________________________________________________________________
def demodulacion_ofdm(simbolo_con_cp, indices_fft, n_fft, n_cp, n_sc):
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]                  # Descarta el prefijo cíclico
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)       # FFT (con normalización inversa)
    return espectro[indices_fft]                               # Devuelve solo las subportadoras activas


# _____________________________________________________________________________________
#  ECUALIZACIÓN ZERO-FORCING                                                  Recepción
#  Entran las subportadoras recibidas Y[k] y el canal H[k]; salen las subportadoras
#  corregidas dividiendo por el canal (X_est = Y / H).
# _____________________________________________________________________________________
def ecualizar_zf(Y, H):
    return Y / H


# _____________________________________________________________________________________
#  QUITAR PILOTOS (extraer datos)                                            Recepción
#  Entra la rejilla ecualizada y salen solo las subportadoras de datos (se descartan
#  las posiciones de pilotos).
# _____________________________________________________________________________________
def extraer_datos(rejilla, n_sc, paso=12):
    _, datos = indices_pilotos_datos(n_sc, paso)   # Posiciones de datos
    return rejilla[datos]


# _____________________________________________________________________________________
#  DEMAPEO QPSK                                                               Recepción
#  Entran símbolos QPSK ruidosos y salen los bits decididos por signo (decisión dura).
# _____________________________________________________________________________________
def demapear_qpsk(simbolos):
    s = simbolos * np.sqrt(2)                     # Deshace la normalización (vuelve a ~{±1})
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
    s = simbolos * np.sqrt(10)                       # Deshace la normalización
    bi = _decidir_pam(np.real(s), [-3, -1, 1, 3], _GRAY_4_INV, 2)  # Eje I → 2 bits
    bq = _decidir_pam(np.imag(s), [-3, -1, 1, 3], _GRAY_4_INV, 2)  # Eje Q → 2 bits
    out = np.empty(4 * len(simbolos), dtype=np.uint8)
    out[0::4], out[1::4] = bi[:, 0], bi[:, 1]
    out[2::4], out[3::4] = bq[:, 0], bq[:, 1]
    return out


_GRAY_8_INV = {v: k for k, v in _GRAY_8.items()}   # Nivel → 3 bits (64-QAM)

# _____________________________________________________________________________________
#  DEMAPEO 64-QAM                                                             Recepción
#  Entran símbolos 64-QAM ruidosos y salen los bits, decidiendo el nivel 8-PAM más
#  cercano en cada eje (I y Q).
# _____________________________________________________________________________________
def demapear_64qam(simbolos):
    s = simbolos * np.sqrt(42)                       # Deshace la normalización
    niveles = [-7, -5, -3, -1, 1, 3, 5, 7]
    bi = _decidir_pam(np.real(s), niveles, _GRAY_8_INV, 3)  # Eje I → 3 bits
    bq = _decidir_pam(np.imag(s), niveles, _GRAY_8_INV, 3)  # Eje Q → 3 bits
    out = np.empty(6 * len(simbolos), dtype=np.uint8)
    out[0::6], out[1::6], out[2::6] = bi[:, 0], bi[:, 1], bi[:, 2]
    out[3::6], out[4::6], out[5::6] = bq[:, 0], bq[:, 1], bq[:, 2]
    return out


# _____________________________________________________________________________________
#  DECISIÓN DURA EN UN EJE PAM (apoyo del demapeo)                            Recepción
#  Entra un eje real, sus niveles válidos y el mapa nivel→bits; sale la matriz de bits
#  eligiendo el nivel más cercano a cada muestra.
# _____________________________________________________________________________________
def _decidir_pam(x, niveles, mapa_inverso, n_bits):
    niveles = np.array(niveles)
    idx = np.argmin(np.abs(x[:, None] - niveles[None, :]), axis=1)  # Nivel más cercano
    bits = np.empty((len(x), n_bits), dtype=np.uint8)
    for k, n in enumerate(niveles[idx]):
        bits[k, :] = mapa_inverso[int(n)]          # Recupera los bits de ese nivel
    return bits


# =====================================================================================
# ||           FLUJO COMPLETO — cómo se encadenan los pasos (un símbolo OFDM)         ||
# =====================================================================================

# _____________________________________________________________________________________
#  CADENA OFDM (TX → CANAL → RX)                                          Todo el flujo
#  Entran símbolos QAM de datos y salen los símbolos de datos recuperados, después de
#  pasar por todos los pasos de transmisión, canal y recepción de un símbolo OFDM.
# _____________________________________________________________________________________
def cadena_ofdm(simbolos_datos, n_sc, n_fft, n_cp, fs, snr_db, rng):
    # --- TRANSMISIÓN ---
    X = insertar_pilotos(simbolos_datos, n_sc)               # Rejilla: datos + pilotos
    _, indices_fft = mapeo_sc_a_fft(X, n_fft)                # Qué bins de la FFT se usan

    # --- CANAL: el canal multiplica cada subportadora por H[k]; el ruido se suma en el tiempo ---
    H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)
    senal_canal, _ = modulacion_ofdm(X * H, n_fft, n_cp)     # IFFT+CP de X·H (señal que llega al RX)
    senal_rx = agregar_ruido_awgn(senal_canal, snr_db, rng)  # AWGN

    # --- RECEPCIÓN ---
    Y = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)   # Quita CP + FFT
    X_est = ecualizar_zf(Y, H)                                        # Divide por H[k]
    return extraer_datos(X_est, n_sc)                                # Quita pilotos → datos
