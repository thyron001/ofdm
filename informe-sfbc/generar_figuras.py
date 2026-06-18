# -*- coding: utf-8 -*-
"""Genera las figuras del informe SFBC con datos reales del simulador."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

sys.path.insert(0, "..")
import app

IMG = "img"
os.makedirs(IMG, exist_ok=True)
C = {"1x1": "#D82A2A", "2x1": "#1E54E0", "2x2": "#18A34B"}
MOD = "16-QAM"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

# Numerologia base
params = app.calcular_parametros_ofdm(5.0, 15.0, "normal", 153600, 4)

# ------------------------------------------------------------------ #
# 1) MONTE CARLO: BER vs SNR (1x1, 2x1, 2x2), 8 corridas, IC 95%
# ------------------------------------------------------------------ #
from scipy.stats import t as t_student
def montecarlo(cfgs, snrs, nbits=40000, nsim=8, seed=2024):
    p = app.calcular_parametros_ofdm(5.0, 15.0, "normal", nbits, 4)
    rng = np.random.default_rng(seed)
    tcrit = float(t_student.ppf(0.975, df=nsim - 1))
    res = {}
    for cfg, (ntx, nrx) in cfgs.items():
        mu, lo, hi = [], [], []
        for snr in snrs:
            bers = []
            for _ in range(nsim):
                bits = rng.integers(0, 2, nbits, dtype=np.uint8)
                bers.append(app.cadena_tx_sfbc_rx(bits, MOD, p, snr, rng, n_tx=ntx, n_rx=nrx)["ber"])
            bers = np.array(bers)
            m = bers.mean(); s = bers.std(ddof=1)
            d = tcrit * s / np.sqrt(nsim)
            mu.append(m); lo.append(max(m - d, 1e-6)); hi.append(m + d)
        res[cfg] = (np.array(mu), np.array(lo), np.array(hi))
    return res

cfgs = {"1x1": (1, 1), "2x1": (2, 1), "2x2": (2, 2)}
snrs = np.arange(0, 19)
print("Corriendo Monte Carlo (puede tardar ~1-2 min)...")
mc = montecarlo(cfgs, snrs)
PISO = 1e-5
labels = {"1x1": "1×1 (sin diversidad)", "2x1": "2×1 (SFBC)", "2x2": "2×2 (SFBC)"}
fig, ax = plt.subplots(figsize=(8, 5.2))
for cfg in ["1x1", "2x1", "2x2"]:
    mu, lo, hi = mc[cfg]
    muc = np.clip(mu, PISO, 1)
    yerr = np.vstack([muc - np.clip(lo, PISO, 1), np.clip(hi, PISO, 1) - muc])
    ax.errorbar(snrs, muc, yerr=yerr, color=C[cfg], marker="o", ms=5, lw=2,
                capsize=3, label=labels[cfg])
ax.set_yscale("log"); ax.set_ylim(PISO, 1); ax.set_xlim(-0.5, 18.5)
ax.set_xlabel("SNR recibida (dB)", fontweight="bold")
ax.set_ylabel("BER (escala logarítmica)", fontweight="bold")
ax.set_title("BER vs SNR — 16-QAM (Monte Carlo, 8 corridas, IC 95%)", fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(loc="lower left", framealpha=0.95)
fig.tight_layout(); fig.savefig(f"{IMG}/montecarlo_sfbc.png", dpi=150); plt.close(fig)
print("  -> montecarlo_sfbc.png")
for cfg in ["1x1", "2x1", "2x2"]:
    print("    MC", cfg, {int(s): round(float(m), 6) for s, m in zip(snrs, mc[cfg][0]) if s in (6, 9, 12, 15, 18)})

# ------------------------------------------------------------------ #
# 2) CONSTELACION: sin diversidad (1x1) vs SFBC (2x2) a 15 dB
# ------------------------------------------------------------------ #
SNR_C = 15
rng = np.random.default_rng(7)
nb = 200000
b = rng.integers(0, 2, nb, dtype=np.uint8)
r1 = app.cadena_tx_sfbc_rx(b, MOD, params, SNR_C, rng, capturar_constelaciones=True, n_tx=1, n_rx=1)
r4 = app.cadena_tx_sfbc_rx(b, MOD, params, SNR_C, rng, capturar_constelaciones=True, n_tx=2, n_rx=2)
ideal = np.unique(np.round(r1["constelacion_tx"], 3))
def sub(arr, n=2500):
    return arr if len(arr) <= n else arr[np.random.default_rng(0).choice(len(arr), n, replace=False)]
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
for ax, res, cfg, col, ttl in [
    (axs[0], r1, "1x1", "#D82A2A", "Sin diversidad (1×1, ZF)"),
    (axs[1], r4, "2x2", "#18A34B", "SFBC 2×2"),
]:
    pts = sub(res["constelacion_rx"])
    ax.scatter(pts.real, pts.imag, s=5, color=col, alpha=0.35, edgecolors="none")
    ax.scatter(ideal.real, ideal.imag, marker="+", s=120, color="black", linewidths=1.6, zorder=5)
    ax.axhline(0, color="#cccccc", lw=0.8); ax.axvline(0, color="#cccccc", lw=0.8)
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_aspect("equal")
    ax.set_title(ttl, fontweight="bold"); ax.set_xlabel("En fase (I)"); ax.set_ylabel("Cuadratura (Q)")
    ax.grid(True, ls=":", alpha=0.4)
fig.suptitle(f"Constelación 16-QAM recibida — SNR {SNR_C} dB", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{IMG}/constelacion_sfbc.png", dpi=150); plt.close(fig)
print("  -> constelacion_sfbc.png")

# ------------------------------------------------------------------ #
# 3,4) TRANSMISION DE IMAGEN (lena) a 12 dB: 1x1, 2x1, 2x2
# ------------------------------------------------------------------ #
SNR_IMG = 12
img = Image.open("../lena.png").convert("RGB").resize((256, 256))
bits, mode, size = app.imagen_a_bits(img)
pimg = app.calcular_parametros_ofdm(5.0, 15.0, "normal", len(bits), 4)
ber_img = {}
rec = {}
rng = np.random.default_rng(101)
for cfg, (ntx, nrx) in cfgs.items():
    res = app.cadena_tx_sfbc_rx(bits, MOD, pimg, SNR_IMG, rng, n_tx=ntx, n_rx=nrx)
    ber_img[cfg] = res["ber"]
    rec[cfg] = app.bits_a_imagen(res["bits_rx"], mode, size)
n_simb = int(np.ceil(len(bits) / 4 / (pimg["n_sc"] // 2 * 2)))
print("  Imagen 256x256, simbolos OFDM:", n_simb)
for cfg in ["1x1", "2x1", "2x2"]:
    print(f"    BER imagen {cfg} @12dB = {ber_img[cfg]:.6f}")

for cfg, etiqueta in [("2x1", "SFBC 2×1"), ("2x2", "SFBC 2×2")]:
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.0))
    axs[0].imshow(img); axs[0].set_title("Original", fontweight="bold"); axs[0].axis("off")
    axs[1].imshow(rec[cfg]); axs[1].set_title(f"Recuperada ({etiqueta})", fontweight="bold"); axs[1].axis("off")
    fig.suptitle(f"Transmisión {etiqueta} — 16-QAM, SNR {SNR_IMG} dB — BER = {ber_img[cfg]*100:.3f} %",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{IMG}/sim_{cfg}_sfbc.png", dpi=150); plt.close(fig)
    print(f"  -> sim_{cfg}_sfbc.png")

# ------------------------------------------------------------------ #
# 5) DIAGRAMA DE BLOQUES
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(13.5, 2.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")
bloques = [
    ("Imagen\n→ Bits", "#EAF0FB", False),
    ("Mapeo\nQAM", "#EAF0FB", False),
    ("Codif. SFBC\n[s₀,s₁ ; −s₁*,s₀*]", "#D7F0DE", True),
    ("2× IFFT\n+ CP", "#EAF0FB", False),
    ("Canales\n+ AWGN", "#FDE8E8", False),
    ("Decod. SFBC\n(Alamouti)", "#D7F0DE", True),
    ("Demapeo\n→ Imagen", "#EAF0FB", False),
]
n = len(bloques); w = 11.5; gap = (100 - n * w) / (n + 1); y = 6; h = 10
xs = []
for i, (txt, color, hl) in enumerate(bloques):
    x = gap + i * (w + gap)
    xs.append((x, x + w))
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                         linewidth=2.2 if hl else 1.3,
                         edgecolor="#18A34B" if hl else "#1E54E0",
                         facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=10.5, fontweight="bold" if hl else "normal")
for i in range(n - 1):
    x0 = xs[i][1]; x1 = xs[i + 1][0]
    ax.add_patch(FancyArrowPatch((x0, y + h / 2), (x1, y + h / 2),
                 arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#444"))
ax.text(50, 20.2, "Cadena de diversidad en transmisión SFBC (Alamouti espacio-frecuencia)",
        ha="center", va="center", fontsize=11.5, fontweight="bold")
ax.text(50, 1.6, "Bloques resaltados (verde) = codificación / decodificación SFBC",
        ha="center", va="center", fontsize=9.5, color="#555")
fig.tight_layout(); fig.savefig(f"{IMG}/diagrama_sfbc.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("  -> diagrama_sfbc.png")

# ------------------------------------------------------------------ #
# 6) TABLA DE PARAMETROS
# ------------------------------------------------------------------ #
filas = [
    ("Ancho de banda (BW)", "5 MHz"),
    ("Espaciamiento (Δf)", "15 kHz"),
    ("Prefijo cíclico", "Normal"),
    ("Modulación", "16-QAM"),
    ("Subportadoras útiles (N_SC)", "300"),
    ("Tamaño de la FFT (N_FFT)", "512"),
    ("Frecuencia de muestreo (fs)", f"{params['fs']/1e6:.3f} MHz"),
    ("Muestras de CP (N_CP)", "36"),
    ("Parejas Alamouti por símbolo (N_SC/2)", "150"),
    ("Configuración SFBC", "2×2 (2 TX, 2 RX)"),
]
fig, ax = plt.subplots(figsize=(7.2, 3.8)); ax.axis("off")
tab = ax.table(cellText=filas, colLabels=["Parámetro", "Valor"],
               cellLoc="left", colLoc="center", loc="center", colWidths=[0.66, 0.34])
tab.auto_set_font_size(False); tab.set_fontsize(10.5); tab.scale(1, 1.5)
for (row, col), cell in tab.get_celld().items():
    cell.set_edgecolor("#cfd8e3")
    if row == 0:
        cell.set_facecolor("#1E54E0"); cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f3f6fb")
fig.suptitle("Parámetros calculados — SFBC", fontweight="bold", y=0.93)
fig.tight_layout(); fig.savefig(f"{IMG}/params_sfbc.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("  -> params_sfbc.png")

print("\nLISTO. BER imagen @12dB:", {k: round(v, 6) for k, v in ber_img.items()})
