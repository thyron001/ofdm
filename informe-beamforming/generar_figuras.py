# -*- coding: utf-8 -*-
"""Genera las figuras del informe de Beamforming (MRT) con datos reales del simulador."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from scipy.stats import t as t_student

sys.path.insert(0, "..")
import app

IMG = "img"
os.makedirs(IMG, exist_ok=True)
MOD = "16-QAM"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

# Colores: ordenes BF y overlay comparativo
C_BF = {"1x1": "#D82A2A", "2x1": "#1E54E0", "4x1": "#18A34B"}
C_OV = {"BF 2x1": "#1E54E0", "SFBC 2x1": "#E58A1A", "MRC 1x2": "#18A34B"}

# Numerologia base
params = app.calcular_parametros_ofdm(5.0, 15.0, "normal", 153600, 4)


def _curva(fn, snrs, nbits, nsim, rng, tcrit):
    """Una curva BER vs SNR con IC 95% (t de Student)."""
    mu, lo, hi = [], [], []
    for snr in snrs:
        bers = np.array([fn(rng.integers(0, 2, nbits, dtype=np.uint8), snr) for _ in range(nsim)])
        m = bers.mean(); s = bers.std(ddof=1)
        d = tcrit * s / np.sqrt(nsim)
        mu.append(m); lo.append(max(m - d, 1e-6)); hi.append(m + d)
    return np.array(mu), np.array(lo), np.array(hi)


snrs = np.arange(0, 19)
NBITS, NSIM = 50000, 8
rng = np.random.default_rng(2024)
tcrit = float(t_student.ppf(0.975, df=NSIM - 1))
PISO = 1e-5


def dibujar(nombre, curvas, etiquetas, colores, titulo):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for clave in curvas:
        mu, lo, hi = curvas[clave]
        muc = np.clip(mu, PISO, 1)
        yerr = np.vstack([muc - np.clip(lo, PISO, 1), np.clip(hi, PISO, 1) - muc])
        ax.errorbar(snrs, muc, yerr=yerr, color=colores[clave], marker="o", ms=5, lw=2,
                    capsize=3, label=etiquetas[clave])
    ax.set_yscale("log"); ax.set_ylim(PISO, 1); ax.set_xlim(-0.5, 18.5)
    ax.set_xlabel("SNR de referencia (dB)", fontweight="bold")
    ax.set_ylabel("BER (escala logarítmica)", fontweight="bold")
    ax.set_title(titulo, fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout(); fig.savefig(f"{IMG}/{nombre}.png", dpi=150); plt.close(fig)
    print(f"  -> {nombre}.png")


# ------------------------------------------------------------------ #
# 1) MONTE CARLO A: ordenes de beamforming 1x1 / 2x1 / 4x1
# ------------------------------------------------------------------ #
print("Corriendo Monte Carlo de ordenes BF (puede tardar)...")
p = app.calcular_parametros_ofdm(5.0, 15.0, "normal", NBITS, 4)
mc_bf = {}
for nombre, ntx in [("1x1", 1), ("2x1", 2), ("4x1", 4)]:
    mc_bf[nombre] = _curva(
        lambda b, s, ntx=ntx: app.cadena_tx_beamforming_rx(b, MOD, p, s, rng, n_tx=ntx)["ber"],
        snrs, NBITS, NSIM, rng, tcrit)
lab_bf = {"1x1": "1×1 (sin beamforming)", "2x1": "2×1 (BF MRT)", "4x1": "4×1 (BF MRT)"}
dibujar("montecarlo_bf", mc_bf, lab_bf, C_BF,
        "BER vs SNR — órdenes de beamforming (16-QAM, MC 8 corridas, IC 95%)")
for k in mc_bf:
    print("   ", k, {int(s): round(float(m), 6) for s, m in zip(snrs, mc_bf[k][0]) if s in (6, 9, 12, 15, 18)})

# ------------------------------------------------------------------ #
# 2) MONTE CARLO B: overlay BF 2x1 / SFBC 2x1 / MRC 1x2 (mismo piso de ruido fijo)
# ------------------------------------------------------------------ #
print("Corriendo Monte Carlo overlay (puede tardar)...")
mc_ov = {}
mc_ov["BF 2x1"] = mc_bf["2x1"]  # reutiliza la curva ya calculada
mc_ov["SFBC 2x1"] = _curva(
    lambda b, s: app.cadena_tx_sfbc_rx(b, MOD, p, s, rng, n_tx=2, n_rx=1, ruido_piso_fijo=True)["ber"],
    snrs, NBITS, NSIM, rng, tcrit)
mc_ov["MRC 1x2"] = _curva(
    lambda b, s: app.cadena_tx_rx(b, MOD, p, s, rng, usar_dft_spread=False, n_rx=2, ruido_piso_fijo=True)["ber"],
    snrs, NBITS, NSIM, rng, tcrit)
lab_ov = {"BF 2x1": "BF 2×1 (MRT, P5)", "SFBC 2x1": "SFBC 2×1 (P4)", "MRC 1x2": "MRC 1×2 (P3)"}
dibujar("montecarlo_overlay", mc_ov, lab_ov, C_OV,
        "BER vs SNR — orden 2: BF y MRC (con CSI) vs SFBC (16-QAM, IC 95%)")
for k in mc_ov:
    print("   ", k, {int(s): round(float(m), 6) for s, m in zip(snrs, mc_ov[k][0]) if s in (6, 9, 12, 15, 18)})

# ------------------------------------------------------------------ #
# 3) CONSTELACION: sin beamforming (1x1) vs BF (4x1) a 15 dB
# ------------------------------------------------------------------ #
SNR_C = 15
rng = np.random.default_rng(7)
nb = 200000
b = rng.integers(0, 2, nb, dtype=np.uint8)
r1 = app.cadena_tx_beamforming_rx(b, MOD, params, SNR_C, rng, capturar_constelaciones=True, n_tx=1)
r4 = app.cadena_tx_beamforming_rx(b, MOD, params, SNR_C, rng, capturar_constelaciones=True, n_tx=4)
ideal = np.unique(np.round(r1["constelacion_tx"], 3))
def sub(arr, n=2500):
    return arr if len(arr) <= n else arr[np.random.default_rng(0).choice(len(arr), n, replace=False)]
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
for ax, res, col, ttl in [
    (axs[0], r1, "#D82A2A", "Sin beamforming (1×1, ZF)"),
    (axs[1], r4, "#18A34B", "Beamforming 4×1 (MRT)"),
]:
    pts = sub(res["constelacion_rx"])
    ax.scatter(pts.real, pts.imag, s=5, color=col, alpha=0.35, edgecolors="none")
    ax.scatter(ideal.real, ideal.imag, marker="+", s=120, color="black", linewidths=1.6, zorder=5)
    ax.axhline(0, color="#cccccc", lw=0.8); ax.axvline(0, color="#cccccc", lw=0.8)
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_aspect("equal")
    ax.set_title(ttl, fontweight="bold"); ax.set_xlabel("En fase (I)"); ax.set_ylabel("Cuadratura (Q)")
    ax.grid(True, ls=":", alpha=0.4)
fig.suptitle(f"Constelación 16-QAM recibida — SNR {SNR_C} dB", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{IMG}/constelacion_bf.png", dpi=150); plt.close(fig)
print("  -> constelacion_bf.png")

# ------------------------------------------------------------------ #
# 4,5) TRANSMISION DE IMAGEN (lena) a 12 dB: 1x1, 2x1, 4x1
# ------------------------------------------------------------------ #
SNR_IMG = 12
img = Image.open("../lena.png").convert("RGB").resize((256, 256))
bits, mode, size = app.imagen_a_bits(img)
pimg = app.calcular_parametros_ofdm(5.0, 15.0, "normal", len(bits), 4)
ber_img, rec = {}, {}
rng = np.random.default_rng(101)
for nombre, ntx in [("1x1", 1), ("2x1", 2), ("4x1", 4)]:
    res = app.cadena_tx_beamforming_rx(bits, MOD, pimg, SNR_IMG, rng, n_tx=ntx)
    ber_img[nombre] = res["ber"]
    rec[nombre] = app.bits_a_imagen(res["bits_rx"], mode, size)
n_simb = int(np.ceil(len(bits) / 4 / pimg["n_sc"]))
print("  Imagen 256x256, simbolos OFDM:", n_simb)
for k in ber_img:
    print(f"    BER imagen {k} @12dB = {ber_img[k]:.6f}")

for cfg, etiqueta in [("2x1", "Beamforming 2×1"), ("4x1", "Beamforming 4×1")]:
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.0))
    axs[0].imshow(img); axs[0].set_title("Original", fontweight="bold"); axs[0].axis("off")
    axs[1].imshow(rec[cfg]); axs[1].set_title(f"Recuperada ({etiqueta})", fontweight="bold"); axs[1].axis("off")
    fig.suptitle(f"Transmisión {etiqueta} — 16-QAM, SNR {SNR_IMG} dB — BER = {ber_img[cfg]*100:.3f} %",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{IMG}/sim_{cfg}_bf.png", dpi=150); plt.close(fig)
    print(f"  -> sim_{cfg}_bf.png")

# ------------------------------------------------------------------ #
# 6) DIAGRAMA DE BLOQUES
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(13.5, 2.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")
bloques = [
    ("Imagen\n→ Bits", "#EAF0FB", False),
    ("Mapeo\nQAM", "#EAF0FB", False),
    ("Precodif. MRT\nw̄ = h̄*/‖h̄‖", "#D7F0DE", True),
    ("Nₜ× IFFT\n+ CP", "#EAF0FB", False),
    ("Canales\n+ AWGN", "#FDE8E8", False),
    ("Detector\nŝ = R/‖h̄‖", "#D7F0DE", True),
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
ax.text(50, 20.2, "Cadena de beamforming en transmisión (precodificación MRT, CSI en el TX)",
        ha="center", va="center", fontsize=11.5, fontweight="bold")
ax.text(50, 1.6, "Bloques resaltados (verde) = precodificación / detección de beamforming",
        ha="center", va="center", fontsize=9.5, color="#555")
fig.tight_layout(); fig.savefig(f"{IMG}/diagrama_bf.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("  -> diagrama_bf.png")

# ------------------------------------------------------------------ #
# 7) TABLA DE PARAMETROS
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
    ("Antenas transmisoras (N_T)", "4"),
    ("Configuración beamforming", "4×1 (4 TX, 1 RX)"),
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
fig.suptitle("Parámetros calculados — Beamforming", fontweight="bold", y=0.93)
fig.tight_layout(); fig.savefig(f"{IMG}/params_bf.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("  -> params_bf.png")

print("\nLISTO. BER imagen @12dB:", {k: round(v, 6) for k, v in ber_img.items()})
