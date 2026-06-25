"""
=====================================================================================
 CODIFICACIÓN DE CANAL LTE — Codificador y Decodificador (3GPP TS 36.212) — solo LEER
=====================================================================================
 Este archivo NO ejecuta nada (no hay servidor, ni Monte Carlo, ni imágenes). Solo
 contiene las funciones de la cadena ordenadas en el orden en que ocurren los pasos:
 TRANSMISIÓN, CANAL y RECEPCIÓN.

 LTE define DOS códigos de canal en el estándar TS 36.212 (sección 5.1.3). Aquí están
 implementados los dos, más el CRC de detección de errores:

   1) CÓDIGO CONVOLUCIONAL "tail-biting", tasa 1/3 (§5.1.3.1)
      · Es el código de los canales de CONTROL de LTE (BCH, DCI…).
      · Registro de 6 memorias (longitud de restricción 7, 64 estados).
      · 3 polinomios generadores: G0=133, G1=171, G2=165 (en octal).
      · "tail-biting": el registro se INICIALIZA con los últimos 6 bits del bloque, de
        modo que el estado inicial y el final coinciden (no se gastan bits de cola).
      · Se decodifica con VITERBI de decisión suave (algoritmo de envoltura, WAVA).

   2) TURBO CÓDIGO, tasa 1/3 (§5.1.3.2)
      · Es el código de los canales de DATOS de LTE (DL-SCH/UL-SCH). Rendimiento
        casi-Shannon. Es el código emblemático de LTE.
      · PCCC: dos codificadores RSC de 8 estados idénticos en paralelo, con un
        ENTRELAZADOR QPP interno entre ellos.
        - g0(D)=1+D²+D³ (realimentación),  g1(D)=1+D+D³ (paridad).
      · Salida: x (sistemático) + z (paridad del 1.º) + z' (paridad del 2.º).
      · Se decodifica con dos SISO MAP (BCJR max-log) que intercambian información
        EXTRÍNSECA durante varias iteraciones (turbo = "realimentación").

   3) CRC-24A (§5.1.1): no corrige, solo DETECTA si el bloque llegó con errores. Es el
      polinomio g(D)=D24+D23+D18+D17+D14+D11+D10+D7+D6+D5+D4+D3+D+1.

 ¿Por qué codificar? (slides "OFDM. Diversidad en f – Codificación de Canal"). Cada bit
 de información se ESPARCE entre varios bits de código; al mapearlos sobre subportadoras
 OFDM bien distribuidas, cada bit experimenta DIVERSIDAD en frecuencia y el receptor
 puede CORREGIR errores. Resultado: para la misma SNR, el BER baja mucho (GANANCIA DE
 CODIFICACIÓN). La corrección real se logra con DECISIÓN SUAVE: en vez de decidir 0/1 en
 el demapeo, se pasa al decodificador un LLR (log-verosimilitud) por bit, ponderado por
 la calidad |H[k]|² de su subportadora; los bits que cayeron en desvanecimientos pesan
 poco y los demás "votan" por ellos.

 Flujo de un bloque:
   bits info → CRC-24A → codificador (conv. o turbo) → QAM → IFFT+CP
        → [canal Pedestrian A + ruido] → quita CP+FFT → ecualización ZF
        → LLR (demapeo suave ponderado por |H|²) → decodificador → verificación CRC → bits
=====================================================================================
"""

import numpy as np


# =====================================================================================
# ||                 PARÁMETROS DEL ESTÁNDAR (TS 36.212)                              ||
# =====================================================================================

# CRC-24A (§5.1.1): posiciones (potencias de D) con coeficiente 1 en g(D).
_CRC24A_POS = [24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0]

# Código convolucional tail-biting (§5.1.3.1): 3 generadores en octal y nº de memorias.
G_CONV = (0o133, 0o171, 0o165)
M_CONV = 6                                  # 6 memorias → 64 estados

# Entrelazador QPP del turbo (Tabla 5.1.3-3): π(i) = (f1·i + f2·i²) mod K. Subconjunto
# de tamaños de bloque K válidos del estándar con sus coeficientes (f1, f2).
QPP = {
    40:   (3,   10),
    512:  (31,  64),
    1024: (31,  84),
    2048: (31,  64),
    6144: (263, 480),
}
K_TURBO = 512                               # Tamaño de bloque turbo usado por defecto


# =====================================================================================
# ||                                TRANSMISIÓN                                       ||
# =====================================================================================

# _____________________________________________________________________________________
#  CRC-24A — ANEXAR                                                          Transmisión
#  Entran los bits de información y salen esos mismos bits con 24 bits de CRC al final.
#  El CRC es el resto de dividir (info·D²⁴) entre el polinomio g(D) en GF(2). Sirve para
#  DETECTAR (no corregir) si el bloque llegó con errores tras decodificar.
# _____________________________________________________________________________________
def _crc24a_resto(bits):
    poly = 0
    for p in _CRC24A_POS:
        poly |= (1 << p)
    poly &= 0xFFFFFF                          # taps por debajo de D²⁴ (los 24 bits bajos)
    reg = 0
    for b in list(int(x) for x in bits) + [0] * 24:   # info seguida de 24 ceros
        msb = (reg >> 23) & 1
        reg = ((reg << 1) | b) & 0xFFFFFF
        if msb:
            reg ^= poly
    return reg                                # resto de 24 bits


def crc24a_anexar(bits):
    resto = _crc24a_resto(bits)
    crc = np.array([(resto >> i) & 1 for i in range(23, -1, -1)], dtype=np.uint8)  # MSB primero
    return np.concatenate([bits.astype(np.uint8), crc])


# _____________________________________________________________________________________
#  CODIFICADOR CONVOLUCIONAL TAIL-BITING (R=1/3, §5.1.3.1)                   Transmisión
#  Entra el bloque de bits y salen 3 bits de código por cada bit de entrada (3 flujos
#  d0,d1,d2). El registro se INICIALIZA con los últimos 6 bits del bloque (tail-biting):
#  así el estado inicial y el final son iguales y no se gastan bits de cola.
# _____________________________________________________________________________________
def _conv_taps():
    # Cada generador octal → 7 taps [entrada, D1, D2, D3, D4, D5, D6] (MSB = entrada).
    return [np.array([(g >> (6 - j)) & 1 for j in range(7)], dtype=np.uint8) for g in G_CONV]


def _conv_paso(estado, u, taps):
    # estado: 6 bits, bit i = contenido de la memoria D_{i+1}. u = bit de entrada.
    Ds = [(estado >> i) & 1 for i in range(6)]            # D1..D6 (contenido actual)
    salidas = []
    for g in taps:
        o = g[0] & u                                      # tap de la entrada
        for i in range(6):
            o ^= g[i + 1] & Ds[i]                         # taps de las memorias
        salidas.append(o & 1)
    nuevo = ((estado << 1) | u) & 0x3F                    # desplaza: D1←u, D2←D1, …
    return nuevo, salidas


def codificar_convolucional(bits):
    bits = bits.astype(np.uint8)
    K = len(bits)
    taps = _conv_taps()
    estado = sum(int(bits[K - 1 - i]) << i for i in range(M_CONV))   # tail-biting: últimos 6 bits
    estado_inicial = estado
    d = np.empty((K, 3), dtype=np.uint8)
    for k in range(K):
        estado, salidas = _conv_paso(estado, int(bits[k]), taps)
        d[k] = salidas
    assert estado == estado_inicial, "tail-biting: el estado final debe igualar al inicial"
    return d.reshape(-1)                                   # 3K bits, intercalados [d0,d1,d2]


# _____________________________________________________________________________________
#  RSC DE 8 ESTADOS (constituyente del turbo) — TRELLIS                      Transmisión
#  No codifica todavía: construye las tablas del codificador recursivo sistemático
#  g0(D)=1+D²+D³ (realimentación), g1(D)=1+D+D³ (paridad). Para cada estado (3 bits) y
#  cada bit de entrada da: estado siguiente, bit sistemático (= entrada) y bit de paridad.
# _____________________________________________________________________________________
def _rsc_trellis():
    nxt = np.zeros((8, 2), dtype=np.int64)
    par = np.zeros((8, 2), dtype=np.int64)
    for s in range(8):
        m1, m2, m3 = s & 1, (s >> 1) & 1, (s >> 2) & 1
        for u in range(2):
            a = u ^ m2 ^ m3                               # realimentación g0=1+D²+D³
            z = a ^ m1 ^ m3                               # paridad g1=1+D+D³
            ns = a | (m1 << 1) | (m2 << 2)               # desplaza el registro
            nxt[s, u] = ns
            par[s, u] = z
    return nxt, par


def _rsc_codificar(bits, trellis):
    nxt, par = trellis
    s = 0                                                 # estado inicial cero
    paridad = np.empty(len(bits), dtype=np.uint8)
    for k, u in enumerate(bits):
        paridad[k] = par[s, u]
        s = nxt[s, u]
    # Terminación de trellis: 3 bits de cola que llevan el registro al estado 0.
    cola_sys = np.empty(3, dtype=np.uint8)
    cola_par = np.empty(3, dtype=np.uint8)
    for j in range(3):
        m1, m2, m3 = s & 1, (s >> 1) & 1, (s >> 2) & 1
        u = m2 ^ m3                                       # entrada que fuerza a=0
        cola_sys[j] = u
        cola_par[j] = par[s, u]
        s = nxt[s, u]
    assert s == 0, "la terminación debe dejar el RSC en el estado 0"
    return paridad, cola_sys, cola_par


# _____________________________________________________________________________________
#  ENTRELAZADOR QPP (interno del turbo, §5.1.3.2.3)                          Transmisión
#  Entra K y sale la permutación π(i)=(f1·i+f2·i²) mod K (los coeficientes vienen de la
#  Tabla 5.1.3-3). Reordena los bits antes del 2.º codificador: es la pieza que hace que
#  los dos codificadores "vean" el bloque de forma distinta (clave del turbo).
# _____________________________________________________________________________________
def qpp_interleaver(K):
    f1, f2 = QPP[K]
    i = np.arange(K)
    perm = (f1 * i + f2 * i * i) % K
    assert len(np.unique(perm)) == K, "el QPP debe ser una permutación válida (biyección)"
    return perm


# _____________________________________________________________________________________
#  CODIFICADOR TURBO (R=1/3, §5.1.3.2)                                       Transmisión
#  Entra un bloque de K bits y salen los flujos del PCCC: sistemático x, paridad z del
#  1.º RSC (sobre los bits en orden) y paridad z' del 2.º RSC (sobre los bits ENTRELAZADOS
#  con QPP), más los bits de cola de terminación de ambos. Tasa ≈ 1/3.
# _____________________________________________________________________________________
def codificar_turbo(bits):
    bits = bits.astype(np.uint8)
    K = len(bits)
    perm = qpp_interleaver(K)
    trellis = _rsc_trellis()
    z1, t1s, t1p = _rsc_codificar(bits, trellis)          # 1.º RSC: bits en orden
    z2, t2s, t2p = _rsc_codificar(bits[perm], trellis)    # 2.º RSC: bits entrelazados
    return {
        "sys": bits, "par1": z1, "par2": z2,
        "t1s": t1s, "t1p": t1p, "t2s": t2s, "t2p": t2p,
        "perm": perm, "K": K,
    }


def turbo_a_vector(c):
    # Serializa los flujos del turbo en un único vector de bits para transmitir.
    return np.concatenate([c["sys"], c["par1"], c["par2"],
                           c["t1s"], c["t1p"], c["t2s"], c["t2p"]]).astype(np.uint8)


# ---- Modulaciones QAM (idénticas a las demás prácticas) -----------------------------

# _____________________________________________________________________________________
#  MAPEO QPSK / 16-QAM                                                       Transmisión
#  Entran bits y salen símbolos del plano I/Q (Gray, energía media = 1).
# _____________________________________________________________________________________
def mapear_qpsk(bits):
    bits = bits.reshape(-1, 2).astype(np.int8)
    i = 1 - 2 * bits[:, 0]
    q = 1 - 2 * bits[:, 1]
    return (i + 1j * q) / np.sqrt(2)


_GRAY_4 = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}     # 2 bits → nivel 4-PAM


def mapear_16qam(bits):
    bits = bits.reshape(-1, 4)
    i = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 0:2]])
    q = np.array([_GRAY_4[(int(a), int(b))] for a, b in bits[:, 2:4]])
    return (i + 1j * q) / np.sqrt(10)


# _____________________________________________________________________________________
#  MODULACIÓN OFDM (IFFT + PREFIJO CÍCLICO)                                  Transmisión
#  Entra la rejilla de subportadoras y sale la señal OFDM en el tiempo con su CP.
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


def modulacion_ofdm(rejilla_sc, n_fft, n_cp):
    vec_freq, indices_fft = mapeo_sc_a_fft(rejilla_sc, n_fft)
    senal_t = np.fft.ifft(vec_freq) * n_fft / np.sqrt(len(rejilla_sc))
    cp = senal_t[-n_cp:] if n_cp > 0 else np.array([], dtype=complex)
    return np.concatenate([cp, senal_t]), indices_fft


# =====================================================================================
# ||                                   CANAL                                          ||
# =====================================================================================

_RETARDOS_PEDA_NS = np.array([0.0, 110.0, 190.0, 410.0])     # Pedestrian A (ITU-R M.1225)
_POTENCIAS_PEDA_DB = np.array([0.0, -9.7, -19.2, -22.8])

# _____________________________________________________________________________________
#  CANAL PEDESTRIAN A                                                              Canal
#  Genera la respuesta en frecuencia H[k] del canal multitrayecto (Rayleigh por tap).
# _____________________________________________________________________________________
def generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng):
    pot = 10 ** (_POTENCIAS_PEDA_DB / 10)
    pot = pot / np.sum(pot)
    a = (rng.standard_normal(len(pot)) + 1j * rng.standard_normal(len(pot))) * np.sqrt(pot / 2)
    retardos = np.round(_RETARDOS_PEDA_NS * 1e-9 * fs).astype(int)
    h = np.zeros(n_fft, dtype=complex)
    for tap, m in enumerate(retardos):
        if m < n_fft:
            h[m] += a[tap]
    return np.fft.fft(h)[indices_fft]


# _____________________________________________________________________________________
#  RUIDO AWGN                                                                      Canal
#  Entra una señal y sale con ruido blanco gaussiano complejo calibrado al SNR (dB).
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
#  Entra la señal recibida y salen las subportadoras (espectro en las posiciones usadas).
# _____________________________________________________________________________________
def demodulacion_ofdm(senal_con_cp, indices_fft, n_fft, n_cp, n_sc):
    sin_cp = senal_con_cp[n_cp:n_cp + n_fft]
    espectro = np.fft.fft(sin_cp) / n_fft * np.sqrt(n_sc)
    return espectro[indices_fft]


# _____________________________________________________________________________________
#  DEMAPEO SUAVE → LLR (la pieza que habilita la corrección de errores)      Recepción
#  Entran los símbolos ECUALIZADOS ŝ[k] y la ganancia de canal g[k]=|H[k]|² de cada
#  subportadora, y salen los LLR (log-verosimilitud) por bit. Convención: LLR > 0 ⇒ el
#  bit es más probablemente 0. La PONDERACIÓN por g[k] es la clave: los bits que cayeron
#  en un desvanecimiento (g pequeña) entran con LLR pequeño (poca confianza) y los demás
#  los corrigen. (Con decisión DURA esta información se perdería.)
# _____________________________________________________________________________________
def llr_qpsk(s_eq, g):
    s = s_eq * np.sqrt(2)                                  # deshace la normalización
    L = np.empty(2 * len(s_eq))
    L[0::2] = g * np.real(s)                               # bit I (signo de la parte real)
    L[1::2] = g * np.imag(s)                               # bit Q (signo de la parte imag.)
    return L


def _llr_eje_4pam(x, g):
    # LLR max-log de los 2 bits de un eje 4-PAM Gray {-3,-1,1,3} (mapa _GRAY_4).
    niveles = np.array([-3, -1, 1, 3])
    b0 = np.array([0, 0, 1, 1])                            # bit alto de cada nivel
    b1 = np.array([0, 1, 1, 0])                            # bit bajo de cada nivel
    d2 = (x[:, None] - niveles[None, :]) ** 2             # distancia² a cada nivel
    L0 = d2[:, b0 == 1].min(axis=1) - d2[:, b0 == 0].min(axis=1)
    L1 = d2[:, b1 == 1].min(axis=1) - d2[:, b1 == 0].min(axis=1)
    return g * L0, g * L1                                  # (×g = ponderación por |H|²)


def llr_16qam(s_eq, g):
    s = s_eq * np.sqrt(10)
    L0i, L1i = _llr_eje_4pam(np.real(s), g)
    L0q, L1q = _llr_eje_4pam(np.imag(s), g)
    L = np.empty(4 * len(s_eq))
    L[0::4], L[1::4] = L0i, L1i                            # 2 bits del eje I
    L[2::4], L[3::4] = L0q, L1q                            # 2 bits del eje Q
    return L


# _____________________________________________________________________________________
#  DECODIFICADOR CONVOLUCIONAL — VITERBI SUAVE TAIL-BITING (WAVA)             Recepción
#  Entran los LLR de los 3 flujos de código (K×3) y salen los K bits decodificados. Para
#  el tail-biting (estado inicial = final, desconocido) se usa el Viterbi de ENVOLTURA:
#  se recorre el bloque circular dos vueltas y se decodifica la 2.ª, ya con las métricas
#  "calentadas". El metric usa los LLR (decisión SUAVE) → de ahí la ganancia.
# _____________________________________________________________________________________
def _conv_trellis_completo():
    taps = _conv_taps()
    nxt = np.zeros((64, 2), dtype=np.int64)
    out = np.zeros((64, 2, 3), dtype=np.int64)
    for s in range(64):
        for u in range(2):
            ns, sal = _conv_paso(s, u, taps)
            nxt[s, u] = ns
            out[s, u] = sal
    return nxt, out


def decodificar_convolucional(LLR, vueltas=4):
    LLR = LLR.reshape(-1, 3)
    K = len(LLR)
    nxt, out = _conv_trellis_completo()
    # Signo de la salida en ±1: (1-2·bit). Métrica de rama = Σ (1-2·bit)·LLR (a MAXIMIZAR).
    signo = 1 - 2 * out                                   # (64,2,3) en {+1,-1}
    # Recorremos el bloque CIRCULAR varias vueltas: la 1.ª "calienta" las métricas para
    # que al llegar a la última vuelta el estado inicial≈final del tail-biting ya esté
    # bien estimado. Decodificamos la ÚLTIMA vuelta (la más fiable).
    LLRv = np.concatenate([LLR] * vueltas)
    T = vueltas * K
    NEG = -1e9
    metrica = np.zeros(64)                                 # todas las arrancadas por igual
    psd = np.zeros((T, 64), dtype=np.int64)               # estado previo (para traceback)
    psu = np.zeros((T, 64), dtype=np.int8)                # bit de entrada (para traceback)
    for t in range(T):
        rama = (signo * LLRv[t][None, None, :]).sum(axis=2)   # (64,2): métrica de cada rama
        nueva = np.full(64, NEG)
        for s in range(64):
            for u in range(2):
                ns = nxt[s, u]
                cand = metrica[s] + rama[s, u]
                if cand > nueva[ns]:
                    nueva[ns] = cand
                    psd[t, ns] = s
                    psu[t, ns] = u
        metrica = nueva
    # Traceback desde el mejor estado final; nos quedamos con la ÚLTIMA vuelta.
    s = int(np.argmax(metrica))
    bits_rev = np.empty(T, dtype=np.uint8)
    for t in range(T - 1, -1, -1):
        bits_rev[t] = psu[t, s]
        s = psd[t, s]
    return bits_rev[(vueltas - 1) * K: vueltas * K]        # los K bits de información


# _____________________________________________________________________________________
#  BCJR max-log (SISO de un RSC) — apoyo del decodificador turbo             Recepción
#  Entran los LLR sistemático y de paridad de un RSC y el LLR a priori (la información que
#  le pasa el OTRO decodificador) y sale el LLR EXTRÍNSECO de cada bit (lo que ESTE
#  decodificador aporta de nuevo). Es media iteración del turbo.
# _____________________________________________________________________________________
def _bcjr_maxlog(Lsys, Lpar, La, term_sys, term_par, trellis):
    nxt, par = trellis
    K = len(Lsys)
    # Secuencias completas: K pasos de datos + 3 de terminación (a priori 0 en la cola).
    Ls = np.concatenate([Lsys, term_sys])
    Lp = np.concatenate([Lpar, term_par])
    Lap = np.concatenate([La, np.zeros(3)])
    T = K + 3
    NEG = -1e9
    # Rama γ por (paso, estado, entrada): ½[(1-2u)(La+Lsys) + (1-2·paridad)Lpar].
    su = (1 - 2 * np.arange(2))                            # ±1 según la entrada u
    alpha = np.full((T + 1, 8), NEG); alpha[0, 0] = 0.0    # arranca en estado 0
    beta = np.full((T + 1, 8), NEG); beta[T, 0] = 0.0      # termina en estado 0
    gamma = np.zeros((T, 8, 2))
    for k in range(T):
        for s in range(8):
            for u in range(2):
                sp = 1 - 2 * par[s, u]
                gamma[k, s, u] = 0.5 * (su[u] * (Lap[k] + Ls[k]) + sp * Lp[k])
    # Recursión hacia adelante (α) y hacia atrás (β), en max-log (suma-máximo).
    for k in range(T):
        nueva = np.full(8, NEG)
        for s in range(8):
            if alpha[k, s] <= NEG / 2:
                continue
            for u in range(2):
                ns = nxt[s, u]
                v = alpha[k, s] + gamma[k, s, u]
                if v > nueva[ns]:
                    nueva[ns] = v
        alpha[k + 1] = nueva
    for k in range(T - 1, -1, -1):
        nueva = np.full(8, NEG)
        for s in range(8):
            for u in range(2):
                ns = nxt[s, u]
                v = beta[k + 1, ns] + gamma[k, s, u]
                if v > nueva[s]:
                    nueva[s] = v
        beta[k] = nueva
    # LLR de salida en los K pasos de datos y extrínseco = salida − a priori − canal.
    Le = np.empty(K)
    for k in range(K):
        m = [NEG, NEG]                                     # mejor métrica con u=0 y u=1
        for s in range(8):
            for u in range(2):
                ns = nxt[s, u]
                v = alpha[k, s] + gamma[k, s, u] + beta[k + 1, ns]
                if v > m[u]:
                    m[u] = v
        Lout = m[0] - m[1]
        Le[k] = Lout - La[k] - Lsys[k]
    return Le


# _____________________________________________________________________________________
#  DECODIFICADOR TURBO (BCJR iterativo)                                       Recepción
#  Entran los LLR sistemático, de paridad 1, de paridad 2 y de las colas, y salen los K
#  bits. Los dos SISO se turnan pasándose información EXTRÍNSECA a través del entrelazador
#  QPP; con cada iteración la estimación mejora (de ahí "turbo"). Decisión final por signo.
# _____________________________________________________________________________________
def decodificar_turbo(Lsys, Lpar1, Lpar2, Lt1s, Lt1p, Lt2s, Lt2p, perm, K, n_iter=6):
    trellis = _rsc_trellis()
    invperm = np.empty(K, dtype=np.int64); invperm[perm] = np.arange(K)
    La1 = np.zeros(K)                                      # a priori del 1.º decodificador
    Le1 = np.zeros(K)
    for _ in range(n_iter):
        Le1 = _bcjr_maxlog(Lsys, Lpar1, La1, Lt1s, Lt1p, trellis)          # 1.º RSC
        La2 = Le1[perm]                                                    # entrelaza para el 2.º
        Le2 = _bcjr_maxlog(Lsys[perm], Lpar2, La2, Lt2s, Lt2p, trellis)    # 2.º RSC
        La1 = Le2[invperm]                                                 # desentrelaza para el 1.º
    L_total = Lsys + Le1 + La1                             # canal + ambos extrínsecos
    return (L_total < 0).astype(np.uint8)                 # LLR<0 ⇒ bit 1


# _____________________________________________________________________________________
#  VERIFICACIÓN CRC-24A                                                       Recepción
#  Entra el bloque decodificado (info + 24 bits de CRC) y dice si el CRC cuadra (True) o
#  si se detectó un error residual (False). Devuelve también los bits de información.
# _____________________________________________________________________________________
def crc24a_verificar(bits):
    info, crc_rx = bits[:-24], bits[-24:]
    resto = _crc24a_resto(info)
    crc_calc = np.array([(resto >> i) & 1 for i in range(23, -1, -1)], dtype=np.uint8)
    ok = bool(np.array_equal(crc_calc, crc_rx))
    return info, ok


# =====================================================================================
# ||   FLUJO COMPLETO — cómo se encadenan los pasos (un bloque de código)             ||
# =====================================================================================

# _____________________________________________________________________________________
#  CADENA DE CODIFICACIÓN (un bloque)                                     Todo el flujo
#  Entran los bits de información de UN bloque y el código elegido ("convolucional" o
#  "turbo"); salen los bits de información recuperados y si el CRC cuadró. Recorre:
#  CRC → codificar → QAM → OFDM → canal+ruido → OFDM⁻¹ → ZF → LLR(|H|²) → decodificar → CRC.
# _____________________________________________________________________________________
def cadena_codificacion(bits_info, modulacion, codigo, n_sc, n_fft, n_cp, fs, snr_db, rng):
    # --- TRANSMISIÓN: CRC + codificación de canal ---
    bloque = crc24a_anexar(bits_info)                     # info + CRC-24A
    if codigo == "convolucional":
        bits_cod = codificar_convolucional(bloque)
    else:                                                 # "turbo"
        c = codificar_turbo(bloque)
        bits_cod = turbo_a_vector(c)

    # --- TRANSMISIÓN: QAM + OFDM (se rellena hasta llenar subportadoras de datos) ---
    bps = 2 if modulacion == "QPSK" else 4
    falta = (-len(bits_cod)) % bps
    if falta:
        bits_cod = np.concatenate([bits_cod, np.zeros(falta, dtype=np.uint8)])
    mapear = mapear_qpsk if modulacion == "QPSK" else mapear_16qam
    simbolos = mapear(bits_cod)
    falta_sc = (-len(simbolos)) % n_sc
    if falta_sc:
        simbolos = np.concatenate([simbolos, np.zeros(falta_sc, dtype=complex)])
    n_ofdm = len(simbolos) // n_sc

    # --- CANAL + RECEPCIÓN por símbolo OFDM: cada símbolo ve un canal independiente ---
    s_eq_total, g_total = [], []
    for i in range(n_ofdm):
        rejilla = simbolos[i * n_sc:(i + 1) * n_sc]
        _, indices_fft = mapeo_sc_a_fft(rejilla, n_fft)
        H = generar_canal_pedestrian_a(n_fft, fs, indices_fft, rng)
        senal_canal, _ = modulacion_ofdm(rejilla * H, n_fft, n_cp)        # Y = H·X
        senal_rx = agregar_ruido_awgn(senal_canal, snr_db, rng)
        rejilla_rx = demodulacion_ofdm(senal_rx, indices_fft, n_fft, n_cp, n_sc)
        s_eq_total.append(rejilla_rx / H)                 # ecualización Zero-Forcing
        g_total.append(np.abs(H) ** 2)                    # calidad de cada subportadora
    s_eq = np.concatenate(s_eq_total)
    g = np.concatenate(g_total)

    # --- RECEPCIÓN: demapeo SUAVE → LLR ponderados por |H|² ---
    llr = (llr_qpsk if modulacion == "QPSK" else llr_16qam)(s_eq, g)

    # --- RECEPCIÓN: decodificación de canal + verificación de CRC ---
    if codigo == "convolucional":
        n_llr = 3 * len(bloque)
        bloque_rx = decodificar_convolucional(llr[:n_llr])
    else:
        K = len(bloque)
        # Recupera los tramos de LLR en el mismo orden que turbo_a_vector().
        o = 0
        Lsys = llr[o:o + K]; o += K
        Lp1 = llr[o:o + K];  o += K
        Lp2 = llr[o:o + K];  o += K
        Lt1s = llr[o:o + 3]; o += 3
        Lt1p = llr[o:o + 3]; o += 3
        Lt2s = llr[o:o + 3]; o += 3
        Lt2p = llr[o:o + 3]; o += 3
        bloque_rx = decodificar_turbo(Lsys, Lp1, Lp2, Lt1s, Lt1p, Lt2s, Lt2p, c["perm"], K)

    info_rx, crc_ok = crc24a_verificar(bloque_rx)
    return info_rx, crc_ok
