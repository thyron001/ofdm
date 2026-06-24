"""
=====================================================================================
 BEAMFORMING — Diversidad/Ganancia en TRANSMISIÓN (precodificación MRT) — solo LEER
=====================================================================================
 Este archivo NO ejecuta nada (no hay servidor, ni Monte Carlo, ni imágenes). Solo
 contiene las funciones de la cadena, ordenadas en el orden en que ocurren los pasos:
 TRANSMISIÓN, CANAL y RECEPCIÓN.

 Beamforming en TX usa N_T antenas TRANSMISORAS y, a diferencia de SFBC, el transmisor
 CONOCE el canal (CSI en TX). Con ese conocimiento multiplica cada subportadora por un
 VECTOR DE PRECODIFICACIÓN (caso de "baja correlación" entre antenas):

        w̄[k] = h̄[k]* / ‖h̄[k]‖          (MRT, Maximum Ratio Transmission)

 donde h̄[k] = [h_1[k], …, h_{N_T}[k]] son los canales de las N_T antenas en la
 subportadora k. El vector w̄:
   · ROTA LA FASE de cada antena para que las señales lleguen ALINEADAS al receptor
     (se suman coherentemente, no se cancelan), y
   · reparte más potencia a las antenas con mejor |h_i|, manteniendo ‖w̄‖=1 (la
     potencia total de transmisión es constante).

 Tras el canal, el receptor de 1 antena ve:
        r[k] = Σ_i h_i[k]·w_i[k]·s[k] + ruido = ‖h̄[k]‖·s[k] + ruido
 así que basta dividir por ‖h̄[k]‖ para recuperar s[k]. La SNR efectiva queda
 multiplicada por ‖h̄[k]‖² = Σ_i |h_i[k]|², lo que da:
   · GANANCIA DE ARREGLO: la potencia útil crece en promedio ×N_T, y
   · DIVERSIDAD de orden N_T: hace falta que TODAS las antenas se desvanezcan a la vez
     para perder la señal (cada vez más improbable al subir N_T).

 Es el DUAL EXACTO de MRC (Práctica 3, diversidad en RX): MRC combina en el receptor con
 conj(H); MRT precodifica en el transmisor con conj(H) normalizado. En OFDM la
 precodificación es POR SUBPORTADORA (cada subportadora ve un canal plano).

 Flujo:
   bits → QAM → s[k] (todas las subportadoras) → precodificar MRT (N_T rejillas)
        → N_T×(IFFT+CP) → [N_T canales se suman en el aire + ruido] → FFT
        → detector (÷‖h̄‖) → QAM⁻¹ → bits
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
    bits = bits.reshape(-1, 2).astype(np.int8)
    i = 1 - 2 * bits[:, 0]
    q = 1 - 2 * bits[:, 1]
    return (i + 1j * q) / np.sqrt(2)


# Mapa Gray de 4-PAM (un eje de 16-QAM): 2 bits → nivel de amplitud
_GRAY_4 = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}

# _____________________________________________________________________________________
#  MAPEO 16-QAM                                                              Transmisión
#  Entran bits y salen símbolos 16-QAM (Gray, energía media = 1). 4 bits/símbolo.
# _____________________________________________________________________________________
def mapear_16qam(bits):
    bits = bits.reshape(-1, 4)
    i = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 0:2]])
    q = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 2:4]])
    return (i + 1j * q) / np.sqrt(10)


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
    bits = bits.reshape(-1, 6)
    i = np.array([_GRAY_8[(int(a), int(b), int(c))] for a, b, c in bits[:, 0:3]])
    q = np.array([_GRAY_8[(int(a), int(b), int(c))] for a, b, c in bits[:, 3:6]])
    return (i + 1j * q) / np.sqrt(42)


# _____________________________________________________________________________________
#  PRECODIFICAR MRT (EL PASO CLAVE EN EL TX)                                 Transmisión
#  Entran los símbolos s_grid (uno por subportadora) y la matriz de canal H (N_T × N_SC,
#  una fila por antena TX) y salen las N_T rejillas a transmitir X (N_T × N_SC) y la
#  ganancia efectiva del enlace 'norma' (N_SC):
#       norma[k] = ‖h̄[k]‖ = sqrt(Σ_i |H_i[k]|²)
#       w_i[k]   = conj(H_i[k]) / norma[k]      (cada columna tiene norma 1 → potencia cte.)
#       X_i[k]   = w_i[k] · s_grid[k]
#  Como w_i = conj(H_i)/‖h̄‖, al pasar por el canal cada antena aporta
#  H_i·w_i = |H_i|²/‖h̄‖, y la SUMA de las N_T antenas da ‖h̄‖·s (suman EN FASE).
# _____________________________________________________________________________________
def precodificar_mrt(s_grid, H):
    norma = np.sqrt(np.sum(np.abs(H) ** 2, axis=0))   # ‖h̄[k]‖ por subportadora
    W = np.conj(H) / (norma + 1e-12)                   # Pesos MRT (cada columna de norma 1)
    X = W * s_grid[np.newaxis, :]                       # Rejilla por antena: X_i[k] = w_i[k]·s[k]
    return X, norma


# _____________________________________________________________________________________
#  MAPEO SUBPORTADORAS → BINS DE LA FFT                                     Transmisión
#  Entra una rejilla de N_SC subportadoras y sale el vector de N_FFT puntos en banda
#  base (DC vacía, centrado en ±frecuencia) y los índices FFT usados.
# _____________________________________________________________________________________
def mapeo_sc_a_fft(rejilla_sc, n_fft):
    n_sc = len(rejilla_sc)
    mitad = n_sc // 2
    indices_pos = np.arange(1, mitad + 1)
    indices_neg = n_fft - np.arange(1, mitad + 1)[::-1]
    indices_fft = np.concatenate([indices_neg, indices_pos])
    vec = np.zeros(n_fft, dtype=complex)
    vec[indices_fft] = rejilla_sc
    return vec, indices_fft


# _____________________________________________________________________________________
#  MODULACIÓN OFDM (IFFT + PREFIJO CÍCLICO)                                  Transmisión
#  Entra la rejilla de una antena y sale su señal OFDM en el tiempo con el prefijo
#  cíclico (CP), junto con los índices FFT usados. (Se aplica a cada antena por separado.)
# _____________________________________________________________________________________
def modulacion_ofdm(rejilla_sc, n_fft, n_cp):
    vec_freq, indices_fft = mapeo_sc_a_fft(rejilla_sc, n_fft)
    senal_t = np.fft.ifft(vec_freq) * n_fft / np.sqrt(len(rejilla_sc))
    cp = senal_t[-n_cp:] if n_cp > 0 else np.array([], dtype=complex)
    return np.concatenate([cp, senal_t]), indices_fft


# =====================================================================================
# ||                                   CANAL                                          ||
# =====================================================================================

# Perfil potencia-retardo del canal Pedestrian A (ITU-R M.1225): 4 trayectorias (taps)
_RETARDOS_PEDA_NS = np.array([0.0, 110.0, 190.0, 410.0])     # Retardo de cada tap (ns)
_POTENCIAS_PEDA_DB = np.array([0.0, -9.7, -19.2, -22.8])     # Potencia relativa de cada tap (dB)

# _____________________________________________________________________________________
#  CANAL PEDESTRIAN A                                                              Canal
#  No entra señal: genera la respuesta en frecuencia H[k] del canal multitrayecto
#  (Rayleigh por tap). Cada antena TX tiene su propio canal independiente hacia el RX.
# _____________________________________________________________________________________
def generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng):
    pot_lin = 10 ** (_POTENCIAS_PEDA_DB / 10)
    pot_lin = pot_lin / np.sum(pot_lin)
    a = (rng.standard_normal(len(pot_lin)) + 1j * rng.standard_normal(len(pot_lin))) \
        * np.sqrt(pot_lin / 2)
    retardos_muestras = np.round(_RETARDOS_PEDA_NS * 1e-9 * fs).astype(int)
    h = np.zeros(n_fft, dtype=complex)
    for tap, m in enumerate(retardos_muestras):
        if m < n_fft:
            h[m] += a[tap]
    H = np.fft.fft(h)
    return H[indices_fft]


# _____________________________________________________________________________________
#  RUIDO AWGN                                                                      Canal
#  Entra una señal y sale con ruido blanco gaussiano complejo, calibrado al SNR (dB).
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
#  Entra la señal recibida por la antena (suma coherente de las N_T antenas TX a través
#  de sus canales) y salen sus N_SC subportadoras: se descarta el CP y se aplica la FFT.
# _____________________________________________________________________________________
def demodulacion_ofdm(simbolo_con_cp, indices_fft, n_fft, n_cp, n_sc):
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)
    return espectro[indices_fft]


# _____________________________________________________________________________________
#  DETECTOR DE BEAMFORMING (EL PASO CLAVE EN EL RX)                           Recepción
#  Entra la rejilla recibida R (N_SC) y la ganancia efectiva 'norma' (N_SC) y sale la
#  estimación del símbolo: ŝ[k] = R[k] / ‖h̄[k]‖. Como el precodificador YA alineó las
#  fases, no queda fase residual que deshacer: basta normalizar por ‖h̄[k]‖.
# _____________________________________________________________________________________
def detector_beamforming(R, norma):
    return R / (norma + 1e-12)


# _____________________________________________________________________________________
#  DEMAPEO QPSK                                                               Recepción
#  Entran símbolos QPSK ruidosos y salen los bits decididos por signo (decisión dura).
# _____________________________________________________________________________________
def demapear_qpsk(simbolos):
    s = simbolos * np.sqrt(2)
    b0 = (np.real(s) < 0).astype(np.uint8)
    b1 = (np.imag(s) < 0).astype(np.uint8)
    out = np.empty(2 * len(simbolos), dtype=np.uint8)
    out[0::2] = b0
    out[1::2] = b1
    return out


_GRAY_4_INV = {v: k for k, v in _GRAY_4.items()}   # Nivel → par de bits (16-QAM)

# _____________________________________________________________________________________
#  DEMAPEO 16-QAM                                                             Recepción
#  Entran símbolos 16-QAM ruidosos y salen los bits (nivel 4-PAM más cercano por eje).
# _____________________________________________________________________________________
def demapear_16qam(simbolos):
    s = simbolos * np.sqrt(10)
    niveles = [-3, -1, 1, 3]
    bi = _decidir_pam(np.real(s), niveles, _GRAY_4_INV, 2)
    bq = _decidir_pam(np.imag(s), niveles, _GRAY_4_INV, 2)
    out = np.empty(4 * len(simbolos), dtype=np.uint8)
    out[0::4], out[1::4] = bi[:, 0], bi[:, 1]
    out[2::4], out[3::4] = bq[:, 0], bq[:, 1]
    return out


_GRAY_8_INV = {v: k for k, v in _GRAY_8.items()}   # Nivel → 3 bits (64-QAM)

# _____________________________________________________________________________________
#  DEMAPEO 64-QAM                                                             Recepción
#  Entran símbolos 64-QAM ruidosos y salen los bits (nivel 8-PAM más cercano por eje).
# _____________________________________________________________________________________
def demapear_64qam(simbolos):
    s = simbolos * np.sqrt(42)
    niveles = [-7, -5, -3, -1, 1, 3, 5, 7]
    bi = _decidir_pam(np.real(s), niveles, _GRAY_8_INV, 3)
    bq = _decidir_pam(np.imag(s), niveles, _GRAY_8_INV, 3)
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
    idx = np.argmin(np.abs(x[:, None] - niveles[None, :]), axis=1)
    bits = np.empty((len(x), n_bits), dtype=np.uint8)
    for k, n in enumerate(niveles[idx]):
        bits[k, :] = mapa_inverso[int(n)]
    return bits


# =====================================================================================
# ||   FLUJO COMPLETO — cómo se encadenan los pasos (un símbolo, N_T TX × 1 RX)       ||
# =====================================================================================

# _____________________________________________________________________________________
#  CADENA BEAMFORMING (N_T antenas TX → 1 antena RX)                      Todo el flujo
#  Entran símbolos QAM de datos (uno por subportadora) y salen recuperados. El TX conoce
#  el canal de las N_T antenas y precodifica con MRT; el RX ve la suma coherente de las
#  N_T antenas y solo tiene que normalizar por ‖h̄‖.
# _____________________________________________________________________________________
def cadena_beamforming(simbolos_datos, n_sc, n_fft, n_cp, fs, snr_db, rng, n_tx):
    # --- CANAL (CSI conocido en el TX): un enlace Pedestrian A por antena ---
    _, indices_fft = mapeo_sc_a_fft(simbolos_datos, n_fft)            # Bins de la FFT en uso
    H = np.stack([generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)
                  for _ in range(n_tx)])                              # H: N_T × N_SC

    # --- TRANSMISIÓN ---
    X, norma = precodificar_mrt(simbolos_datos, H)                    # ← PASO CLAVE: N_T rejillas MRT

    # --- CANAL + RECEPCIÓN: la antena RX ve la SUMA coherente de las N_T antenas ---
    senal_rx = sum(modulacion_ofdm(X[a] * H[a], n_fft, n_cp)[0]       # Cada antena por su canal…
                   for a in range(n_tx))                              # …y se suman EN FASE en el aire
    senal_rx = agregar_ruido_awgn(senal_rx, snr_db, rng)             # Ruido en el receptor
    R = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)  # Quita CP + FFT

    # --- DETECCIÓN ---
    return detector_beamforming(R, norma)                            # ← PASO CLAVE: ŝ = R/‖h̄‖
