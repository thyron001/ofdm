"""
=====================================================================================
 MRC — Diversidad en RECEPCIÓN (Maximal Ratio Combining) — solo para LEER y entender
=====================================================================================
 Este archivo NO ejecuta nada (no hay servidor, ni Monte Carlo, ni imágenes). Solo
 contiene las funciones de la cadena, ordenadas en el orden en que ocurren los pasos:
 TRANSMISIÓN, CANAL y RECEPCIÓN.

 La transmisión es OFDM normal. LO NUEVO está en la RECEPCIÓN: el mismo símbolo OFDM
 llega por n_rx antenas, cada una con su propio canal H_m[k] independiente y su propio
 ruido. La recepción tiene DOS pasos que conviene NO confundir:

   1) Combinación MRC (diapositivas 122-126). Según la teoría, el peso óptimo de cada
      antena es el conjugado de su canal, w_m = conj(H_m), porque w_MRC = h es lo que
      maximiza la SNR tras la combinación. Con esos pesos las ramas se suman de forma
      COHERENTE (se alinean en fase) y cada una pesa según su ganancia:

              z[k] = Σ_m conj(H_m[k]) · Y_m[k]                  (esto es lo que dicen las diapositivas)

   2) Normalización a la escala del símbolo (estimación de s, diapositiva 125). La suma
      anterior vale z = (Σ_m |H_m|²)·X + ruido, o sea X escalado por la ganancia combinada.
      Para poder DECIDIR los bits contra la constelación QAM de referencia se divide por
      esa misma ganancia y así se recupera la estimación del símbolo transmitido:

              X_est[k] = z[k] / Σ_m |H_m[k]|²

      Con 1 antena este segundo paso es la ecualización Zero-Forcing (Y/H). La división es
      un escalar real positivo: no cambia la SNR ni el orden de diversidad, solo reescala.

 Con más antenas sube la SNR efectiva (∝ Σ|H_m|², la suma de las SNR de cada rama) y baja
 la tasa de error: es la ganancia de diversidad de orden N_R (diapositiva 127).

 Flujo:
   bits → QAM → pilotos → IFFT+CP → [ n_rx canales + ruido ] → FFT → MRC → quitar pilotos → QAM → bits
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
#  POSICIONES DE PILOTOS Y DATOS                                             Transmisión
#  Entra el nº de subportadoras y salen los índices de pilotos y de datos.
# _____________________________________________________________________________________
def indices_pilotos_datos(n_sc, paso=12):
    indices = np.arange(n_sc)
    pilotos = indices[::paso]                                     # 1 piloto cada `paso` (12)
    datos = np.setdiff1d(indices, pilotos, assume_unique=False)
    return pilotos, datos


# _____________________________________________________________________________________
#  INSERTAR PILOTOS (armar la rejilla)                                      Transmisión
#  Entran los símbolos QAM y sale la rejilla de n_sc subportadoras con los pilotos
#  (valor conocido 1+0j) intercalados cada 12 posiciones.
# _____________________________________________________________________________________
def insertar_pilotos(simbolos_datos, n_sc, paso=12, valor_piloto=1 + 0j):
    pilotos, datos = indices_pilotos_datos(n_sc, paso)
    rejilla = np.zeros(n_sc, dtype=complex)
    rejilla[pilotos] = valor_piloto
    n = min(len(datos), len(simbolos_datos))
    rejilla[datos[:n]] = simbolos_datos[:n]
    return rejilla


# _____________________________________________________________________________________
#  MAPEO SUBPORTADORAS → BINS DE LA FFT                                     Transmisión
#  Entra la rejilla de n_sc subportadoras y sale el vector de n_fft puntos en banda
#  base (DC vacía, subportadoras centradas en ±frecuencia) y los índices FFT usados.
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
#  Entra la rejilla de subportadoras y sale la señal OFDM en el tiempo con el prefijo
#  cíclico (CP) antepuesto, junto con los índices FFT usados.
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
#  (Rayleigh por tap). Como es aleatorio, cada antena RX recibe un canal distinto.
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
#  Cada antena RX recibe su propio ruido independiente.
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
#  Entra la señal recibida por UNA antena (en el tiempo) y salen sus n_sc subportadoras
#  (en frecuencia): se descarta el CP y se aplica la FFT.
# _____________________________________________________________________________________
def demodulacion_ofdm(simbolo_con_cp, indices_fft, n_fft, n_cp, n_sc):
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)
    return espectro[indices_fft]


# _____________________________________________________________________________________
#  COMBINACIÓN MRC + NORMALIZACIÓN (EL PASO CLAVE DE LA DIVERSIDAD EN RX)     Recepción
#  Entra una lista con un par (Y_m, H_m) por cada antena receptora. Son DOS pasos:
#
#    Paso 1 - MRC propiamente dicho (diapositivas 122-126). El peso de cada antena es
#             w_m = conj(H_m), así que la combinación coherente es la suma ponderada:
#                   z = Σ_m conj(H_m)·Y_m
#             Esa suma vale (Σ|H_m|²)·X + ruido, es decir, X ESCALADO por la ganancia;
#             todavía NO es la estimación del símbolo.
#
#    Paso 2 - Normalización (diapositiva 125): se divide z por Σ|H_m|² para devolver la
#             estimación de X en su escala original, de modo que el demapeo QAM (que usa
#             umbrales fijos) decida bien. Con una sola antena esto es Zero-Forcing (Y/H).
# _____________________________________________________________________________________
def combinar_mrc(recibidos):
    n_sc = len(recibidos[0][0])
    num = np.zeros(n_sc, dtype=complex)      # Paso 1: z = Σ conj(H_m)·Y_m  (combinación MRC)
    den = np.zeros(n_sc, dtype=float)        #         Σ |H_m|²  (ganancia combinada del canal)
    for Y_m, H_m in recibidos:
        num += np.conj(H_m) * Y_m            # peso conj(H_m): alinea la fase y pondera por la ganancia
        den += np.abs(H_m) ** 2              # acumula |H_m|² de todas las antenas

    # Paso 2: divide por la ganancia combinada para devolver X_est en la escala del símbolo.
    # El "+ 1e-12" es solo un SEGURO NUMÉRICO: si en una subportadora TODAS las antenas
    # caen a la vez en un desvanecimiento profundo, Σ|H_m|² ≈ 0 y la división daría inf/NaN.
    # Ese valor es tan pequeño que no afecta el resultado en las subportadoras normales;
    # solo evita el "dividir entre cero" en el caso extremo.
    return num / (den + 1e-12)


# _____________________________________________________________________________________
#  QUITAR PILOTOS (extraer datos)                                            Recepción
#  Entra la rejilla combinada y salen solo las subportadoras de datos.
# _____________________________________________________________________________________
def extraer_datos(rejilla, n_sc, paso=12):
    _, datos = indices_pilotos_datos(n_sc, paso)
    return rejilla[datos]


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
# ||      FLUJO COMPLETO — cómo se encadenan los pasos (un símbolo, n_rx antenas)     ||
# =====================================================================================

# _____________________________________________________________________________________
#  CADENA MRC (TX OFDM → n_rx CANALES → RX con MRC)                       Todo el flujo
#  Entran símbolos QAM de datos y salen recuperados. El mismo símbolo OFDM se manda por
#  n_rx antenas (canal y ruido independientes en cada una) y se combinan con MRC.
# _____________________________________________________________________________________
def cadena_mrc(simbolos_datos, n_sc, n_fft, n_cp, fs, snr_db, rng, n_rx):
    # --- TRANSMISIÓN (una sola vez) ---
    X = insertar_pilotos(simbolos_datos, n_sc)               # Rejilla: datos + pilotos
    _, indices_fft = mapeo_sc_a_fft(X, n_fft)                # Qué bins de la FFT se usan

    # --- CANAL + RECEPCIÓN por antena: cada antena ve un canal y un ruido distintos ---
    recibidos = []
    for _ in range(n_rx):
        H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)   # Canal de esta antena
        senal_canal, _ = modulacion_ofdm(X * H, n_fft, n_cp)          # IFFT+CP de X·H
        senal_rx = agregar_ruido_awgn(senal_canal, snr_db, rng)       # Ruido de esta antena
        Y = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)  # Quita CP + FFT
        recibidos.append((Y, H))                                      # Guarda (Y_m, H_m)

    # --- COMBINACIÓN ---
    X_est = combinar_mrc(recibidos)              # ← PASO CLAVE: junta todas las antenas (MRC)
    return extraer_datos(X_est, n_sc)            # Quita pilotos → datos
