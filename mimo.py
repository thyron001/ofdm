"""
=====================================================================================
 MIMO — DIVERSIDAD COMBINADA TX + RX (SFBC + MRC) — solo para LEER y entender
=====================================================================================
 Este archivo NO ejecuta nada (no hay servidor, ni Monte Carlo, ni imágenes). Solo
 contiene las funciones de la cadena, ordenadas en el orden en que ocurren los pasos:
 TRANSMISIÓN, CANAL y RECEPCIÓN.

 Esta práctica MEZCLA las dos técnicas de diversidad ya vistas para conseguir la MÁXIMA
 diversidad de un sistema MIMO, enviando la MISMA información (de forma redundante) por
 TODAS las antenas — NO flujos independientes:

   · DIVERSIDAD EN TX  → SFBC (código de Alamouti, Práctica 4): 2 antenas TRANSMISORAS
     envían, por cada par de subportadoras (k, k+1), los símbolos (s0, s1) y su versión
     ortogonal (-s1*, s0*). Cada antena se escala por 1/√2 (reparte la potencia total).
   · DIVERSIDAD EN RX  → MRC (Maximal Ratio Combining, Práctica 3): las N_R antenas
     RECEPTORAS se combinan sumando coherentemente, ponderando por la ganancia de canal.

 El receptor hace las dos cosas a la vez: por cada antena RX aplica el combinador
 ortogonal de Alamouti (recupera la diversidad TX) y SUMA las N_R antenas ponderando por
 Σ(|h1|²+|h2|²) (eso es MRC → recupera la diversidad RX):

     s0_est = Σ_rx (conj(h1)·r0 + h2·conj(r1)) / Σ_rx (|h1|² + |h2|²)
     s1_est = Σ_rx (conj(h1)·r1 − h2·conj(r0)) / Σ_rx (|h1|² + |h2|²)
              └──────── combinación Alamouti (TX) ───────┘   └──── suma MRC (RX) ────┘

 Resultado: la ORDEN DE DIVERSIDAD es el PRODUCTO N_T·N_R (2 TX × N_R RX):
     2×1 → orden 2,   2×2 → orden 4,   2×4 → orden 8.
 La curva BER vs SNR cae con MAYOR PENDIENTE al subir la orden de diversidad.

 Flujo:
   bits → QAM → pares (s0,s1) → Alamouti (2 antenas TX) → IFFT+CP
        → [2 canales por cada antena RX + ruido] → FFT
        → combinador Alamouti + suma MRC (sobre las N_R antenas) → rearmar pares → QAM → bits
=====================================================================================
"""

import numpy as np

ESC = 1.0 / np.sqrt(2.0)   # Reparto de potencia entre las 2 antenas transmisoras (1/√2)


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
#  SEPARAR EN PARES (s0, s1)                                                 Transmisión
#  Entra el bloque de símbolos QAM y salen dos vectores: s0 (posición par → subportadora
#  k) y s1 (posición impar → subportadora k+1). Ambos símbolos se transmitirán de forma
#  REDUNDANTE por las 2 antenas (la misma información, no flujos distintos).
# _____________________________________________________________________________________
def separar_pares(simbolos):
    s0 = simbolos[0::2]
    s1 = simbolos[1::2]
    return s0, s1


# _____________________________________________________________________________________
#  CODIFICAR ALAMOUTI — DIVERSIDAD EN TX (EL PASO CLAVE DEL TX)              Transmisión
#  Entran los pares (s0, s1) y salen las DOS rejillas de subportadoras, una por antena TX:
#       Antena 1:  k → s0 ,   k+1 → s1
#       Antena 2:  k → -s1*,  k+1 → s0*
#  Ambas escaladas por 1/√2 para repartir la potencia total. Como las 2 subportadoras del
#  par son adyacentes, H[k] ≈ H[k+1] y el combinador ortogonal recupera s0 y s1 sin
#  interferencia, ganando diversidad de TX sin conocer el canal en el transmisor.
# _____________________________________________________________________________________
def codificar_alamouti(s0, s1, n_sc, esc=ESC):
    n_par = n_sc // 2
    x1 = np.zeros(n_sc, dtype=complex)
    x2 = np.zeros(n_sc, dtype=complex)
    x1[0::2][:n_par] = esc * s0
    x1[1::2][:n_par] = esc * s1
    x2[0::2][:n_par] = esc * (-np.conj(s1))
    x2[1::2][:n_par] = esc * (np.conj(s0))
    return x1, x2


# _____________________________________________________________________________________
#  MAPEO SUBPORTADORAS → BINS DE LA FFT                                     Transmisión
#  Entra una rejilla de n_sc subportadoras y sale el vector de n_fft puntos en banda
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
#  (Rayleigh por tap). Cada enlace antena-TX → antena-RX tiene su propio canal.
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
#  Entra la señal recibida por UNA antena (suma de las 2 antenas TX a través del canal)
#  y salen sus n_sc subportadoras: se descarta el CP y se aplica la FFT.
# _____________________________________________________________________________________
def demodulacion_ofdm(simbolo_con_cp, indices_fft, n_fft, n_cp, n_sc):
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)
    return espectro[indices_fft]


# _____________________________________________________________________________________
#  COMBINADOR ALAMOUTI + MRC — DIVERSIDAD TX y RX (EL PASO CLAVE DEL RX)      Recepción
#  Entra una lista con una tupla (r0, r1, h1, h2) por CADA antena receptora, donde r0,r1
#  son lo recibido en el par (k, k+1) y h1,h2 los canales de TX1, TX2 hacia esa antena RX.
#  Salen s0_est y s1_est. Hay DOS niveles de diversidad combinados en una sola fórmula:
#     · Dentro de cada antena RX, el combinador ORTOGONAL DE ALAMOUTI separa s0 y s1
#       (diversidad en TX, orden 2).
#     · La SUMA sobre las N_R antenas RX pondera por Σ(|h1|²+|h2|²): eso es MRC
#       (diversidad en RX, orden N_R).
#  Orden de diversidad total = 2·N_R. El /esc final deshace el reparto de potencia 1/√2.
# _____________________________________________________________________________________
def combinar_alamouti_mrc(recibidos, esc=ESC):
    n_par = len(recibidos[0][0])
    num0 = np.zeros(n_par, dtype=complex)    # Acumulador del estimador de s0
    num1 = np.zeros(n_par, dtype=complex)    # Acumulador del estimador de s1
    den = np.zeros(n_par, dtype=float)       # Σ (|h1|²+|h2|²) sobre las antenas RX (parte MRC)
    for r0, r1, h1, h2 in recibidos:          # ← SUMA sobre antenas RX = MRC (diversidad RX)
        num0 += np.conj(h1) * r0 + h2 * np.conj(r1)   # ← combinación Alamouti (diversidad TX)
        num1 += np.conj(h1) * r1 - h2 * np.conj(r0)
        den += np.abs(h1) ** 2 + np.abs(h2) ** 2
    s0_est = (num0 / (den + 1e-12)) / esc
    s1_est = (num1 / (den + 1e-12)) / esc
    return s0_est, s1_est


# _____________________________________________________________________________________
#  REARMAR PARES                                                              Recepción
#  Entran las estimaciones s0_est y s1_est y sale el bloque de símbolos intercalado en el
#  orden original (par/impar), listo para demapear.
# _____________________________________________________________________________________
def rearmar_pares(s0_est, s1_est):
    n_datos = len(s0_est) + len(s1_est)
    datos = np.empty(n_datos, dtype=complex)
    datos[0::2] = s0_est
    datos[1::2] = s1_est
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


# =====================================================================================
# ||   FLUJO COMPLETO — cómo se encadenan los pasos (un símbolo, 2 TX × N_R RX)       ||
# =====================================================================================

# _____________________________________________________________________________________
#  CADENA MIMO DIVERSIDAD COMBINADA (2 TX Alamouti → N_R RX con MRC)      Todo el flujo
#  Entran símbolos QAM de datos (cantidad par) y salen recuperados. El TX envía la misma
#  información redundante con Alamouti (2 antenas), cada antena RX ve la suma de las 2
#  antenas TX por canales independientes, y el receptor combina Alamouti (TX) + MRC (RX).
#  La orden de diversidad resultante es 2·N_R.
# _____________________________________________________________________________________
def cadena_mimo_div(simbolos_datos, n_sc, n_fft, n_cp, fs, snr_db, rng, n_rx):
    n_par = n_sc // 2

    # --- TRANSMISIÓN: misma información, codificada con Alamouti sobre 2 antenas TX ---
    s0, s1 = separar_pares(simbolos_datos)                   # Pares (s0, s1)
    X1, X2 = codificar_alamouti(s0, s1, n_sc)               # ← DIVERSIDAD TX: 2 rejillas Alamouti
    _, indices_fft = mapeo_sc_a_fft(X1, n_fft)              # Qué bins de la FFT se usan

    # --- CANAL + RECEPCIÓN por antena RX: cada una ve la suma de TX1 y TX2 por su canal ---
    recibidos = []
    for _ in range(n_rx):                                    # ← DIVERSIDAD RX: N_R antenas
        H1 = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)   # Canal TX1 → esta RX
        H2 = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)   # Canal TX2 → esta RX
        senal1, _ = modulacion_ofdm(X1 * H1, n_fft, n_cp)             # Aporte de la antena 1
        senal2, _ = modulacion_ofdm(X2 * H2, n_fft, n_cp)             # Aporte de la antena 2
        senal_rx = agregar_ruido_awgn(senal1 + senal2, snr_db, rng)   # Se suman en el aire + AWGN
        R = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)  # Quita CP + FFT
        r0 = R[0::2][:n_par]      # Recibido en la subportadora k
        r1 = R[1::2][:n_par]      # Recibido en la subportadora k+1
        h1 = H1[0::2][:n_par]     # Canal TX1 en el par (H1[k] ≈ H1[k+1])
        h2 = H2[0::2][:n_par]     # Canal TX2 en el par
        recibidos.append((r0, r1, h1, h2))

    # --- COMBINACIÓN: Alamouti (diversidad TX) + MRC sobre las N_R antenas (diversidad RX) ---
    s0_est, s1_est = combinar_alamouti_mrc(recibidos)       # ← PASO CLAVE: orden de diversidad 2·N_R
    return rearmar_pares(s0_est, s1_est)                    # Rearma el bloque de datos
