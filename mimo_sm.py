"""
=====================================================================================
 MIMO — MULTIPLEXACIÓN ESPACIAL (Spatial Multiplexing) — solo para LEER y entender
=====================================================================================
 Este archivo NO ejecuta nada (no hay servidor, ni Monte Carlo, ni imágenes). Solo
 contiene las funciones de la cadena, ordenadas en el orden en que ocurren los pasos:
 TRANSMISIÓN, CANAL y RECEPCIÓN.

 Multiplexación espacial usa N_T antenas TRANSMISORAS y N_R antenas RECEPTORAS para
 enviar FLUJOS DE DATOS COMPLETAMENTE INDEPENDIENTES de forma simultánea, uno por
 antena TX. A diferencia de la diversidad (P3-MRC, P4-SFBC) y el beamforming (P5-MRT),
 que usan las antenas para ROBUSTECER UN SOLO flujo, aquí las antenas MULTIPLICAN el
 número de flujos: con N_L = min(N_T, N_R) flujos se transmite N_L veces más tasa.

 Modelo por subportadora (desvanecimiento plano en cada subportadora OFDM):

        r̄[k] = H[k] · s̄[k] + n̄[k]

 donde s̄[k] = [s_1[k], …, s_{N_T}[k]] son los N_T símbolos simultáneos (uno por antena),
 H[k] es la matriz de canal N_R × N_T y r̄[k] lo recibido por las N_R antenas. Cada antena
 TX se escala por 1/√N_T para repartir la potencia total (potencia radiada constante), lo
 que cuesta ~10·log10(N_T) dB por flujo: ese es el precio de duplicar la tasa.

 El receptor SEPARA los flujos. Se implementan tres detectores:
   · Zero-Forcing (ZF):  ŝ = H⁺·r   (pseudoinversa). Simple, pero REALZA EL RUIDO cuando
     H está mal condicionada (det(H) pequeño → casi singular).
   · MMSE:  ŝ = (HᴴH + (1/SNR)·I)⁻¹ Hᴴ r. Equilibra supresión de interferencia y realce
     de ruido; a SNR alta tiende a ZF, a SNR baja lo mejora.
   · Maximum-Likelihood (ML):  ŝ = argmin ‖r − H·s‖² sobre toda la constelación. Óptimo en
     BER (no invierte H), pero su costo crece como M^{N_T}.

 Flujo:
   bits → QAM → demux en N_T flujos → mapear (×1/√N_T) → N_T×(IFFT+CP)
        → [N_T canales por antena RX + ruido] → FFT → detector (ZF/MMSE/ML)
        → remux → QAM⁻¹ → bits
=====================================================================================
"""

import itertools

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
#  DEMULTIPLEXAR EN N_T FLUJOS (EL PASO CLAVE EN EL TX)                      Transmisión
#  Entra el bloque de símbolos QAM (N_T·N_SC símbolos) y salen N_T flujos independientes,
#  uno por antena. El símbolo i va a la antena (i mod N_T): así cada antena lleva su
#  propio sub-stream de datos (a diferencia de SFBC, donde las antenas llevan versiones
#  redundantes del MISMO par de símbolos).
# _____________________________________________________________________________________
def demux_flujos(simbolos, n_tx):
    return [simbolos[a::n_tx] for a in range(n_tx)]   # n_tx flujos de N_SC símbolos c/u


# _____________________________________________________________________________________
#  MAPEAR FLUJOS A LAS REJILLAS DE ANTENA                                    Transmisión
#  Entran los N_T flujos (N_SC símbolos c/u) y sale la matriz X (N_T × N_SC): la rejilla
#  que transmite cada antena, escalada por esc = 1/√N_T para repartir la potencia total
#  entre las N_T antenas (potencia radiada constante = la de un SISO a plena potencia).
# _____________________________________________________________________________________
def mapear_mimo(flujos, esc):
    return np.stack([esc * f for f in flujos])        # X: N_T × N_SC


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
#  CANAL PEDESTRIAN A (un enlace)                                                  Canal
#  No entra señal: genera la respuesta en frecuencia H[k] del canal multitrayecto
#  (Rayleigh por tap) de UN enlace antena-TX → antena-RX.
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
#  MATRIZ DE CANAL MIMO H[k] (N_R × N_T por subportadora)                          Canal
#  Genera un enlace Pedestrian A INDEPENDIENTE para cada par (antena RX, antena TX). La
#  baja correlación entre enlaces es justo lo que permite separar los flujos. Devuelve un
#  tensor H de forma (N_R, N_T, N_SC): para la subportadora k, H[:, :, k] es la matriz 2×2
#  (en 2×2) del modelo r̄[k] = H[k]·s̄[k] + n̄[k].
# _____________________________________________________________________________________
def generar_canal_mimo(n_fft, fs, indices_fft, rng, n_tx, n_rx):
    n_sc = len(indices_fft)
    H = np.zeros((n_rx, n_tx, n_sc), dtype=complex)
    for r in range(n_rx):
        for t in range(n_tx):
            H[r, t] = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)  # enlace TX t → RX r
    return H


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
#  Entra la señal recibida por una antena (suma de las N_T antenas TX a través del canal)
#  y salen sus N_SC subportadoras: se descarta el CP y se aplica la FFT.
# _____________________________________________________________________________________
def demodulacion_ofdm(simbolo_con_cp, indices_fft, n_fft, n_cp, n_sc):
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)
    return espectro[indices_fft]


# _____________________________________________________________________________________
#  DETECTOR ZERO-FORCING (EL PASO CLAVE EN EL RX)                             Recepción
#  Entra el vector recibido r̄ (N_R) y la matriz de canal EFECTIVA Hg = esc·H (N_R × N_T),
#  que ya incluye el reparto de potencia 1/√N_T. Sale la estimación ŝ (N_T):
#        ŝ = H_g⁺ · r̄        (H_g⁺ = pseudoinversa de Moore-Penrose; = H_g⁻¹ si es cuadrada)
#  Como ŝ = s̄ + H_g⁺·n̄, si H está mal condicionada (casi singular) H_g⁺ AMPLIFICA el ruido.
#  Las filas de H_g⁺ implementan IRC (rechazo de interferencia entre flujos).
# _____________________________________________________________________________________
def detector_zf(r_k, Hg_k):
    return np.linalg.pinv(Hg_k) @ r_k


# _____________________________________________________________________________________
#  DETECTOR MMSE                                                              Recepción
#  Igual entrada que ZF más el cargado de ruido. Sale ŝ con el equilibrio óptimo (en MSE)
#  entre suprimir interferencia y no realzar el ruido:
#        ŝ = (H_gᴴ H_g + (N0/Es)·I)⁻¹ H_gᴴ · r̄
#  El término (N0/Es)·I regulariza la inversión: a SNR alta → tiende a ZF; a SNR baja →
#  frena el realce de ruido a costa de algo de interferencia residual. OJO: con la
#  normalización OFDM el cargado N0/Es NO es 1/SNR sino ≈ (N_SC/N_FFT)/SNR (la potencia de
#  ruido por subportadora se reescala en la FFT); si se usara 1/SNR el MMSE quedaría
#  sobre-regularizado y podría incluso quedar peor que ZF a SNR alta.
# _____________________________________________________________________________________
def detector_mmse(r_k, Hg_k, carga_ruido):
    n_tx = Hg_k.shape[1]
    A = Hg_k.conj().T @ Hg_k + carga_ruido * np.eye(n_tx)
    return np.linalg.solve(A, Hg_k.conj().T @ r_k)


# _____________________________________________________________________________________
#  DETECTOR MAXIMUM-LIKELIHOOD (ML)                                           Recepción
#  Entra r̄, la matriz efectiva H_g y la lista de TODAS las combinaciones candidatas de
#  símbolos (M^{N_T} vectores de N_T símbolos ideales). Sale el ŝ que MINIMIZA la distancia:
#        ŝ = argmin_{s̄ ∈ candidatos} ‖ r̄ − H_g · s̄ ‖²
#  Óptimo en BER porque NO invierte H (no realza ruido) y evalúa la geometría conjunta,
#  pero su costo crece como M^{N_T} (por eso en la práctica se limita a QPSK/16-QAM).
# _____________________________________________________________________________________
def detector_ml(r_k, Hg_k, combinaciones):
    pred = Hg_k @ combinaciones.T                       # (N_R, M^{N_T}): r̄ esperado por candidato
    dist = np.sum(np.abs(pred - r_k[:, None]) ** 2, axis=0)
    return combinaciones[np.argmin(dist)]


# _____________________________________________________________________________________
#  REMULTIPLEXAR LOS FLUJOS                                                   Recepción
#  Entra la matriz de flujos estimados S_est (N_T × N_SC) y sale el bloque de símbolos en
#  el orden original (intercalado por antena), listo para demapear.
# _____________________________________________________________________________________
def remux_flujos(S_est):
    n_tx, n_sc = S_est.shape
    datos = np.empty(n_tx * n_sc, dtype=complex)
    for a in range(n_tx):
        datos[a::n_tx] = S_est[a]
    return datos


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


# _____________________________________________________________________________________
#  CONSTELACIÓN COMPLETA Y COMBINACIONES (apoyo del detector ML)              Recepción
#  constelacion_qam: genera los M = 2^bps símbolos ideales de la modulación (todos los
#  patrones de bits). combinaciones: el producto cartesiano de N_T constelaciones, es
#  decir TODOS los vectores s̄ candidatos que el ML compara (M^{N_T} filas × N_T columnas).
# _____________________________________________________________________________________
def constelacion_qam(mapear, bps):
    M = 1 << bps
    idx = np.arange(M)
    bits = ((idx[:, None] >> np.arange(bps - 1, -1, -1)[None, :]) & 1).astype(np.uint8).reshape(-1)
    return mapear(bits)


def combinaciones(constelacion, n_tx):
    return np.array(list(itertools.product(constelacion, repeat=n_tx)))


# =====================================================================================
# ||   FLUJO COMPLETO — cómo se encadenan los pasos (un símbolo, N_T TX × N_R RX)     ||
# =====================================================================================

# _____________________________________________________________________________________
#  CADENA MULTIPLEXACIÓN ESPACIAL (N_T TX → N_R RX)                       Todo el flujo
#  Entran N_T·N_SC símbolos QAM (N_T flujos de N_SC símbolos) y salen recuperados. Se
#  demultiplexan en N_T flujos (uno por antena), cada antena RX ve la SUMA de las N_T
#  antenas TX por su propio canal, y el detector elegido (ZF/MMSE/ML) separa los flujos
#  subportadora a subportadora.
# _____________________________________________________________________________________
def cadena_mimo_sm(simbolos_datos, n_sc, n_fft, n_cp, fs, snr_db, rng,
                   n_tx, n_rx, receptor="zf", combos=None):
    esc = 1.0 / np.sqrt(n_tx)                                  # Reparto de potencia 1/√N_T

    # --- TRANSMISIÓN ---
    flujos = demux_flujos(simbolos_datos, n_tx)               # ← PASO CLAVE: N_T flujos independientes
    X = mapear_mimo(flujos, esc)                              # Rejilla por antena (N_T × N_SC)
    _, indices_fft = mapeo_sc_a_fft(X[0], n_fft)             # Qué bins de la FFT se usan

    # --- CANAL: matriz H[k] (N_R × N_T) por subportadora ---
    H = generar_canal_mimo(n_fft, fs, indices_fft, rng, n_tx, n_rx)

    # --- RECEPCIÓN por antena: cada RX ve la SUMA de las N_T antenas TX + ruido ---
    R = np.zeros((n_rx, n_sc), dtype=complex)
    pot_rx = 0.0                                              # Potencia recibida (para el cargado del MMSE)
    for r in range(n_rx):
        senal = sum(modulacion_ofdm(X[t] * H[r, t], n_fft, n_cp)[0]   # antena t por su canal H[r,t]
                    for t in range(n_tx))                            # …todas suman en el aire
        pot_rx += np.mean(np.abs(senal) ** 2)                        # potencia recibida (sin ruido)
        senal = agregar_ruido_awgn(senal, snr_db, rng)               # AWGN en el receptor
        R[r] = demodulacion_ofdm(senal, indices_fft, n_fft, n_cp, n_sc)

    # --- DETECCIÓN subportadora a subportadora: separa los N_T flujos ---
    snr_lin = 10 ** (snr_db / 10)
    carga_ruido = (pot_rx / n_rx) * (n_sc / n_fft) / snr_lin  # N0/Es por subportadora (normalización OFDM)
    S_est = np.zeros((n_tx, n_sc), dtype=complex)
    for k in range(n_sc):
        Hg_k = esc * H[:, :, k]                               # Canal efectivo s̄ → r̄ (incluye 1/√N_T)
        r_k = R[:, k]
        if receptor == "zf":
            S_est[:, k] = detector_zf(r_k, Hg_k)              # ← PASO CLAVE RX (ZF)
        elif receptor == "mmse":
            S_est[:, k] = detector_mmse(r_k, Hg_k, carga_ruido)  # ← PASO CLAVE RX (MMSE)
        else:
            S_est[:, k] = detector_ml(r_k, Hg_k, combos)      # ← PASO CLAVE RX (ML)

    return remux_flujos(S_est)                               # Rearma el bloque de datos
