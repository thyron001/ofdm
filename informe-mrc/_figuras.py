# Genera las figuras del informe de la Practica 3 (Diversidad en RX con MRC) a partir
# de simulaciones reales con la imagen lena.png. Reutiliza la cadena de app.py.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import t as t_student
import app

IMGDIR = "informe-mrc/img"
COL = {2: "#18A34B", 4: "#1E54E0", 8: "#D82A2A"}   # mismos colores que la interfaz
BW, DF, CP = 5.0, 15.0, "normal"

# --- Imagen de prueba: Lena reescalada para que la simulacion sea agil ---
lena = Image.open("lena.png").convert("RGB").resize((96, 96), Image.LANCZOS)
bits_img, modo, size = app.imagen_a_bits(lena)
print(f"Lena 96x96 RGB -> {len(bits_img)} bits")

# =====================================================================
# FIGURA 1: recuperacion de la imagen con 2, 4 y 8 antenas (16-QAM)
# =====================================================================
SNR_IMG = 6.0
MOD_IMG = "16-QAM"
params_img = app.calcular_parametros_ofdm(BW, DF, CP, len(bits_img), app.MODULACIONES[MOD_IMG]["bits"])
fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
axes[0].imshow(lena); axes[0].set_title("Original", fontsize=11, weight="bold"); axes[0].axis("off")
ber_img = {}
for ax, n_rx in zip(axes[1:], [2, 4, 8]):
    rng = np.random.default_rng(100 + n_rx)
    r = app.cadena_tx_rx(bits_img, MOD_IMG, params_img, SNR_IMG, rng,
                         capturar_constelaciones=False, usar_dft_spread=False, n_rx=n_rx)
    img_rx = app.bits_a_imagen(r["bits_rx"], modo, size)
    ber_img[n_rx] = r["ber"]
    ax.imshow(img_rx)
    ax.set_title(f"{n_rx} antenas (MRC)\nBER = {r['ber']:.2e}", fontsize=10, weight="bold")
    ax.axis("off")
fig.suptitle(f"Recuperacion de Lena con MRC — 16-QAM, SNR = {SNR_IMG:.0f} dB por antena",
             fontsize=12, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{IMGDIR}/lena_mrc.png", dpi=150, bbox_inches="tight")
plt.close()
print("FIG1 lena_mrc.png  BER:", {k: f"{v:.2e}" for k, v in ber_img.items()})

# =====================================================================
# FIGURA 2: constelacion 16-QAM antes (1 antena, ZF) vs despues (MRC)
# =====================================================================
SNR_C, NRX_C = 8.0, 8
rng = np.random.default_rng(7)
bits_c = rng.integers(0, 2, 16000, dtype=np.uint8)
params_c = app.calcular_parametros_ofdm(BW, DF, CP, len(bits_c), 4)
rc = app.cadena_tx_rx(bits_c, "16-QAM", params_c, SNR_C, rng,
                      capturar_constelaciones=True, usar_dft_spread=False, n_rx=NRX_C)
antes = rc["constelacion_rx_antes"]
despues = rc["constelacion_rx"]
def submu(a, n=2500):
    return a if len(a) <= n else a[np.random.default_rng(0).choice(len(a), n, replace=False)]
antes, despues = submu(antes), submu(despues)
niveles = np.array([-3, -1, 1, 3]) / np.sqrt(10)
ideal = np.array([complex(i, q) for i in niveles for q in niveles])
fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
for a, datos, titulo, col in [
    (ax[0], antes, f"Antes de MRC (1 antena, ZF)", "#D82A2A"),
    (ax[1], despues, f"Despues de MRC ({NRX_C} antenas)", "#18A34B")]:
    a.scatter(datos.real, datos.imag, s=4, c=col, alpha=0.35, linewidths=0)
    a.scatter(ideal.real, ideal.imag, marker="x", c="black", s=70, linewidths=2, zorder=5)
    a.set_title(titulo, fontsize=11, weight="bold")
    a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5)
    a.axhline(0, color="#ccc", lw=0.8); a.axvline(0, color="#ccc", lw=0.8)
    a.set_xlabel("En fase (I)"); a.set_ylabel("Cuadratura (Q)")
    a.set_aspect("equal")
fig.suptitle(f"Constelacion 16-QAM recibida — SNR = {SNR_C:.0f} dB por antena",
             fontsize=12, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{IMGDIR}/constelacion_mrc.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"FIG2 constelacion_mrc.png  BER(1 antena ref) implicito; puntos antes={len(antes)} despues={len(despues)}")

# =====================================================================
# FIGURA 3: Monte Carlo BER vs SNR por nº de antenas, para 3 modulaciones
# =====================================================================
SNRS = list(range(0, 16))
N_SIM, NBITS = 10, 15000
ANTENAS = [2, 4, 8]
MODS = ["QPSK", "16-QAM", "64-QAM"]
tcrit = float(t_student.ppf(0.975, df=N_SIM - 1))
rng = np.random.default_rng(2024)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
resumen = {}
for ax, mod in zip(axes, MODS):
    bps = app.MODULACIONES[mod]["bits"]
    params = app.calcular_parametros_ofdm(BW, DF, CP, NBITS, bps)
    for n_rx in ANTENAS:
        mu, lo, hi = [], [], []
        for snr in SNRS:
            bers = []
            for _ in range(N_SIM):
                b = rng.integers(0, 2, NBITS, dtype=np.uint8)
                r = app.cadena_tx_rx(b, mod, params, snr, rng, usar_dft_spread=False, n_rx=n_rx)
                bers.append(r["ber"])
            bers = np.array(bers)
            m = bers.mean(); s = bers.std(ddof=1)
            mar = tcrit * s / np.sqrt(N_SIM)
            mu.append(max(m, 1e-5)); lo.append(max(m - mar, 1e-5)); hi.append(max(m + mar, 1e-5))
        ax.plot(SNRS, mu, "-o", color=COL[n_rx], ms=4, lw=1.8, label=f"{n_rx} antenas")
        ax.fill_between(SNRS, lo, hi, color=COL[n_rx], alpha=0.18)
        resumen[(mod, n_rx)] = mu
    ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
    ax.set_xlim(0, 15); ax.set_xlabel("SNR por antena (dB)")
    ax.set_title(mod, fontsize=12, weight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower left")
axes[0].set_ylabel("BER (escala log)")
fig.suptitle("BER vs SNR con diversidad en RX (MRC) — Monte Carlo, 10 corridas, IC 95%",
             fontsize=12.5, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{IMGDIR}/montecarlo_mrc.png", dpi=150, bbox_inches="tight")
plt.close()
print("FIG3 montecarlo_mrc.png listo")
# Numeros clave para el texto
for mod in MODS:
    fila = " | ".join(f"{n}:{resumen[(mod,n)][8]:.2e}" for n in ANTENAS)
    print(f"  {mod} BER@8dB -> {fila}")
for mod in MODS:
    fila = " | ".join(f"{n}:{resumen[(mod,n)][12]:.2e}" for n in ANTENAS)
    print(f"  {mod} BER@12dB -> {fila}")
print("OK figuras generadas")
