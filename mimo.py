"""
=====================================================================================
 MIMO — MULTIPLEXACIÓN ESPACIAL MULTI-CODEWORD + SIC — solo para LEER y entender
=====================================================================================
 Este archivo NO ejecuta nada (no hay servidor, ni Monte Carlo, ni imágenes). Solo
 contiene las funciones de la cadena, ordenadas en el orden en que ocurren los pasos:
 TRANSMISIÓN, CANAL y RECEPCIÓN.

 MULTIPLEXACIÓN ESPACIAL (Cap. 5, pág. 181-208): con N_T antenas TX y N_R antenas RX se
 transmiten N_L = min(N_T, N_R) SEÑALES COMPLETAMENTE INDEPENDIENTES a la vez, una por
 antena. La tasa se multiplica por N_L (el tiempo de envío se divide), PERO las señales
 se INTERFIEREN mutuamente en el aire: cada antena RX recibe la SUMA de las N_T señales.
 Cuantas más antenas, más interferencia entre señales.

 TX MULTI-CODEWORD (diapositiva 204): cada señal pasa por su PROPIO bloque de
 "Coding & modulation" ANTES del mapeo a antenas. Eso permite dar DIFERENTE ROBUSTEZ a
 cada señal (PARC — Per-Antenna Rate Control, diapositiva 208):
     · La 1ª señal en decodificarse sufre la interferencia de TODAS las demás
       → se le asigna la modulación MÁS ROBUSTA (menor tasa, p.ej. QPSK).
     · Las últimas se decodifican casi sin interferencia
       → pueden usar modulaciones más agresivas (16/64-QAM).

 RX NO LINEAL — SIC, Successive Interference Cancellation (diapositivas 206-207):
 un receptor lineal (ZF/MMSE) deja interferencia residual entre señales. El SIC la
 elimina POR ETAPAS, con un procesamiento NO LINEAL (la decisión dura del demapeo):
     1. Demodula y DECODIFICA la 1ª señal (la más robusta) con un detector MMSE.
     2. La RE-CODIFICA (re-encoding: decisión dura → símbolos limpios) y la RESTA
        de lo recibido por todas las antenas.
     3. La 2ª señal se decodifica ya SIN la interferencia de la 1ª (SIR mejorada),
        se re-codifica y se resta… y así sucesivamente hasta la señal N_L.

 Modelo por subportadora (fading plano en cada subportadora OFDM):
        r̄[k] = H[k] · s̄[k] + n̄[k]
 con s̄[k] los N_T símbolos simultáneos, H[k] la matriz N_R × N_T y r̄[k] lo recibido.
 Cada antena TX se escala por 1/√N_T (potencia total constante).

 Flujo:
   bits → demux en N_L señales → [modulación QAM POR SEÑAL, PARC] → mapeo a antenas
        → N_T×(IFFT+CP) → [canal MIMO H + ruido] → FFT ×N_R
        → SIC: {detectar MMSE → decodificar → re-codificar → restar} ×N_L
        → remux de bits → imagen
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
#  PERFIL PARC (Per-Antenna Rate Control)                                    Transmisión
#  Entra la modulación "objetivo" (la de la señal MENOS interferida) y el nº de señales,
#  y sale la lista de modulaciones por señal, EN EL ORDEN EN QUE SE DECODIFICAN:
#     · señal 0 (se decodifica PRIMERO, sufre interferencia de todas las demás)
#       → modulación más ROBUSTA (escalera descendente hacia QPSK).
#     · señal N_L-1 (se decodifica ÚLTIMA, ya casi sin interferencia gracias al SIC)
#       → la modulación seleccionada (la más agresiva del perfil).
#  Ej.: perfil_parc("64-QAM", 4) → ['QPSK', 'QPSK', '16-QAM', '64-QAM']
# _____________________________________________________________________________________
_ORDEN_MODS = ["QPSK", "16-QAM", "64-QAM"]   # De más robusta a menos robusta

def perfil_parc(modulacion, n_senales):
    idx = _ORDEN_MODS.index(modulacion)
    return [_ORDEN_MODS[max(0, idx - (n_senales - 1 - i))] for i in range(n_senales)]


# _____________________________________________________________________________________
#  DEMULTIPLEXAR LOS BITS EN N_L SEÑALES (diapo 205)                         Transmisión
#  Entra el bloque de bits de UN uso de canal y sale una lista con los bits de cada
#  señal: la señal i (con bps_i bits/símbolo) toma n_sc·bps_i bits consecutivos. Los
#  datos se originan en la MISMA fuente (la imagen) y se demultiplexan en señales
#  distintas ANTES de la modulación de cada una (TX multi-codeword).
# _____________________________________________________________________________________
def demux_bits(bits_uso, n_sc, perfil):
    flujos, pos = [], 0
    for m in perfil:
        bps = MODULACIONES[m]["bits"]
        flujos.append(bits_uso[pos:pos + n_sc * bps])
        pos += n_sc * bps
    return flujos


# _____________________________________________________________________________________
#  "CODING & MODULATION" POR SEÑAL (EL PASO CLAVE DEL TX, diapo 204)         Transmisión
#  Entran los bits de cada señal y su perfil PARC, y sale la matriz S (N_T × N_SC): cada
#  fila es una señal INDEPENDIENTE modulada con SU PROPIA modulación (multi-codeword).
#  A diferencia de SFBC (P4), aquí las antenas NO llevan versiones redundantes: cada
#  una transporta datos distintos → la tasa total se multiplica por N_L.
# _____________________________________________________________________________________
def modular_senales(flujos_bits, perfil, n_sc):
    S = np.zeros((len(perfil), n_sc), dtype=complex)
    for i, m in enumerate(perfil):
        S[i] = MODULACIONES[m]["mapear"](flujos_bits[i])
    return S


# _____________________________________________________________________________________
#  MAPEO A ANTENAS ("Mapping to antennas", diapo 204)                        Transmisión
#  Entra la matriz de señales S y sale X = S/√N_T: la señal i se asigna a la antena i,
#  con reparto de potencia 1/√N_T para que la potencia TOTAL radiada sea constante
#  (la misma que un SISO a plena potencia). Este reparto cuesta ~10·log10(N_T) dB por
#  señal: parte del precio de multiplicar la tasa.
# _____________________________________________________________________________________
def mapear_a_antenas(S, n_tx):
    return S / np.sqrt(n_tx)


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
#  Genera un enlace Pedestrian A INDEPENDIENTE por cada par (antena RX, antena TX): la
#  baja correlación entre enlaces es la condición para poder separar las señales
#  (pág. 181). Devuelve un tensor H de forma (N_R, N_T, N_SC): para la subportadora k,
#  H[:, :, k] es la matriz del modelo r̄[k] = H[k]·s̄[k] + n̄[k].
# _____________________________________________________________________________________
def generar_canal_mimo(n_fft, fs, indices_fft, rng, n_tx, n_rx):
    n_sc = len(indices_fft)
    H = np.zeros((n_rx, n_tx, n_sc), dtype=complex)
    for r in range(n_rx):
        for t in range(n_tx):
            H[r, t] = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)
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
#  Entra la señal recibida por UNA antena (la SUMA de las N_T señales TX, cada una por su
#  canal — ahí está la interferencia) y salen sus n_sc subportadoras.
# _____________________________________________________________________________________
def demodulacion_ofdm(simbolo_con_cp, indices_fft, n_fft, n_cp, n_sc):
    sin_cp = simbolo_con_cp[n_cp:n_cp + n_fft]
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)
    return espectro[indices_fft]


# _____________________________________________________________________________________
#  DETECTOR MMSE DE UNA ETAPA (apoyo del SIC)                                 Recepción
#  Entra lo recibido residual R (N_R × N_SC) y las columnas ACTIVAS del canal efectivo
#  Hg_act (N_SC × N_R × N_ACT): la señal a detectar (columna 0) y las que AÚN interfieren
#  (columnas 1..). Sale la estimación de la señal 0:
#        W = (H_aᴴ H_a + (N0/Es)·I)⁻¹ H_aᴴ           (filtro MMSE por subportadora)
#        ŝ = fila 0 de (W · r̄)
#  El término (N0/Es)·I frena el realce de ruido. OJO: con la normalización OFDM el
#  cargado N0/Es vale (potencia_recibida)·(N_SC/N_FFT)/SNR, no simplemente 1/SNR.
# _____________________________________________________________________________________
def detector_mmse_etapa(R, Hg_act, carga_ruido):
    n_act = Hg_act.shape[2]
    HaH = np.conj(np.transpose(Hg_act, (0, 2, 1)))            # (N_SC, N_ACT, N_R)
    A = HaH @ Hg_act + carga_ruido * np.eye(n_act)            # (N_SC, N_ACT, N_ACT)
    W = np.linalg.solve(A, HaH)                               # (N_SC, N_ACT, N_R)
    rs = np.transpose(R, (1, 0))[:, :, None]                  # (N_SC, N_R, 1)
    return (W @ rs)[:, 0, 0]                                  # ŝ de la señal 0 (N_SC,)


# _____________________________________________________________________________________
#  SIC — CANCELACIÓN SUCESIVA DE INTERFERENCIA (EL PASO CLAVE DEL RX)         Recepción
#  Implementa el receptor NO LINEAL de las diapositivas 206-207. Entra lo recibido R
#  (N_R × N_SC), el canal efectivo Hg (N_R × N_T × N_SC, ya con el 1/√N_T) y el perfil
#  PARC. Para cada señal i = 0 … N_L-1 (¡en orden de robustez!):
#     1. DETECTA la señal i con MMSE, tratando las señales i+1… como interferencia.
#        (La señal 0 sufre N_T-1 interferentes → por eso es la más robusta.)
#     2. "DECODIFICA": demapeo por decisión dura (procesamiento NO LINEAL).
#     3. "RE-ENCODING": vuelve a mapear los bits decididos → réplica limpia de la señal.
#     4. RESTA su contribución (Hg_i · ŝ_limpia) de TODAS las antenas: la señal i+1 se
#        decodificará con SIR mejorada.
#  Si una decisión es errónea la resta introduce error (propagación de errores): otra
#  razón para dar más robustez a las primeras señales.
#  Devuelve (bits por señal, símbolos detectados por señal).
# _____________________________________________________________________________________
def sic_rx(R, Hg, perfil, carga_ruido):
    n_rx, n_tx, n_sc = Hg.shape
    Hg_t = np.transpose(Hg, (2, 0, 1))                        # (N_SC, N_R, N_T)
    R_res = R.copy()                                          # Residual: se le van restando señales
    bits_por_senal, simbolos_por_senal = [], []
    for i in range(n_tx):                                     # ← Etapas del SIC (diapo 207)
        s_est = detector_mmse_etapa(R_res, Hg_t[:, :, i:], carga_ruido)   # 1. Detección MMSE
        mod = MODULACIONES[perfil[i]]
        bits_i = mod["demapear"](s_est)                       # 2. "Decoding" (decisión dura, NO lineal)
        s_limpia = mod["mapear"](bits_i)                      # 3. "Re-encoding" (réplica limpia)
        R_res = R_res - Hg[:, i, :] * s_limpia[None, :]       # 4. Cancela su interferencia
        bits_por_senal.append(bits_i)
        simbolos_por_senal.append(s_est)
    return bits_por_senal, simbolos_por_senal


# _____________________________________________________________________________________
#  RECEPTOR LINEAL (SIN SIC) — solo para comparar                             Recepción
#  Detecta TODAS las señales de una vez con un único filtro MMSE, sin cancelar nada:
#  cada señal queda con la interferencia residual de las demás. Comparado con el SIC
#  evidencia el beneficio del procesamiento no lineal.
# _____________________________________________________________________________________
def rx_lineal(R, Hg, perfil, carga_ruido):
    n_rx, n_tx, n_sc = Hg.shape
    Hg_t = np.transpose(Hg, (2, 0, 1))                        # (N_SC, N_R, N_T)
    HgH = np.conj(np.transpose(Hg_t, (0, 2, 1)))
    A = HgH @ Hg_t + carga_ruido * np.eye(n_tx)
    W = np.linalg.solve(A, HgH)                               # (N_SC, N_T, N_R)
    S = (W @ np.transpose(R, (1, 0))[:, :, None])[:, :, 0]    # (N_SC, N_T)
    bits_por_senal = [MODULACIONES[perfil[i]]["demapear"](S[:, i]) for i in range(n_tx)]
    simbolos_por_senal = [S[:, i] for i in range(n_tx)]
    return bits_por_senal, simbolos_por_senal


# _____________________________________________________________________________________
#  REMULTIPLEXAR LOS BITS                                                     Recepción
#  Entran los bits decodificados de cada señal y sale el bloque en el orden original
#  (el inverso exacto de demux_bits), listo para rearmar la imagen.
# _____________________________________________________________________________________
def remux_bits(bits_por_senal):
    return np.concatenate(bits_por_senal)


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


# Diccionario que asocia cada modulación con sus bits/símbolo y sus funciones de mapeo.
MODULACIONES = {
    "QPSK":   {"bits": 2, "mapear": mapear_qpsk,  "demapear": demapear_qpsk},
    "16-QAM": {"bits": 4, "mapear": mapear_16qam, "demapear": demapear_16qam},
    "64-QAM": {"bits": 6, "mapear": mapear_64qam, "demapear": demapear_64qam},
}


# =====================================================================================
# ||   FLUJO COMPLETO — cómo se encadenan los pasos (un uso de canal, N_T × N_R)      ||
# =====================================================================================

# _____________________________________________________________________________________
#  CADENA MULTIPLEXACIÓN ESPACIAL MULTI-CODEWORD + SIC                    Todo el flujo
#  Entra el bloque de bits de UN uso de canal (n_sc·Σbps bits) y salen los bits
#  recuperados en el mismo orden. Las N_L señales viajan SIMULTÁNEAMENTE (misma banda,
#  mismo tiempo): el tiempo de envío se divide ~N_L, a cambio de interferencia mutua
#  que el SIC va cancelando etapa por etapa.
# _____________________________________________________________________________________
def cadena_mimo_sm(bits_uso, n_sc, n_fft, n_cp, fs, snr_db, rng,
                   n_tx, n_rx, perfil, usar_sic=True):
    # --- TRANSMISIÓN (diapo 204): demux → coding & modulation POR SEÑAL → antenas ---
    flujos_bits = demux_bits(bits_uso, n_sc, perfil)         # N_L señales independientes
    S = modular_senales(flujos_bits, perfil, n_sc)           # ← Multi-codeword (PARC)
    X = mapear_a_antenas(S, n_tx)                            # Mapping to antennas (×1/√N_T)
    _, indices_fft = mapeo_sc_a_fft(X[0], n_fft)             # Bins de la FFT en uso

    # --- CANAL: matriz H[k] por subportadora; cada RX recibe la SUMA de las N_T señales ---
    H = generar_canal_mimo(n_fft, fs, indices_fft, rng, n_tx, n_rx)
    R = np.zeros((n_rx, n_sc), dtype=complex)
    pot_rx = 0.0
    for r in range(n_rx):
        senal = sum(modulacion_ofdm(X[t] * H[r, t], n_fft, n_cp)[0]   # señal t por su canal…
                    for t in range(n_tx))                             # …TODAS suman en el aire
        pot_rx += np.mean(np.abs(senal) ** 2)                         # (para calibrar el MMSE)
        senal = agregar_ruido_awgn(senal, snr_db, rng)                # AWGN en el receptor
        R[r] = demodulacion_ofdm(senal, indices_fft, n_fft, n_cp, n_sc)

    # --- RECEPCIÓN (diapo 207): SIC no lineal (o lineal para comparar) ---
    Hg = H / np.sqrt(n_tx)                                   # Canal efectivo (incluye 1/√N_T)
    snr_lin = 10 ** (snr_db / 10)
    carga = (pot_rx / n_rx) * (n_sc / n_fft) / snr_lin       # N0/Es por subportadora
    if usar_sic:
        bits_por_senal, _ = sic_rx(R, Hg, perfil, carga)     # ← PASO CLAVE: SIC por etapas
    else:
        bits_por_senal, _ = rx_lineal(R, Hg, perfil, carga)  # Lineal (con interferencia residual)
    return remux_bits(bits_por_senal)                        # Bits en el orden original
