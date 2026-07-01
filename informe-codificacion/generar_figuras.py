# Genera las figuras del informe de la Practica 7 (Codificacion de canal LTE:
# codigos convolucionales y turbo codigos) a partir de simulaciones reales.
# Reutiliza la cadena de app.py (misma logica que la pestana del sitio).
# Se ejecuta desde la raiz del repo:  python informe-codificacion/generar_figuras.py
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
import numpy as np
from PIL import Image
from scipy.stats import t as t_student
import app

IMGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(IMGDIR, exist_ok=True)
BW, DF, CP = 5.0, 15.0, "normal"
COL = {"ninguno": "#D82A2A", "convolucional": "#E58A1A", "turbo": "#18A34B"}
ETQ = {"ninguno": "Sin codificar", "convolucional": "Convolucional (Viterbi)", "turbo": "Turbo (BCJR)"}
CODIGOS = ["ninguno", "convolucional", "turbo"]
resultados = {}

LOREM = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec sodales ipsum vel ante "
         "vulputate, id feugiat ligula imperdiet. Duis lacus turpis, pretium non lectus sed, "
         "accumsan tincidunt nisi. In hac habitasse platea dictumst. Ut eu mi augue. Vestibulum "
         "non fringilla elit, a imperdiet libero. Duis porta in sapien sed pretium. Cras id arcu "
         "mauris. Morbi vehicula ligula at diam fringilla, ut malesuada elit semper.\n\n"
         "Sed eget elementum arcu, nec rhoncus ligula. Sed vel magna augue. Morbi vel nunc non "
         "augue lacinia sagittis id nec nibh. Donec in commodo risus. Maecenas dictum, ipsum non "
         "pretium volutpat, orci velit condimentum leo, quis posuere dui sem pulvinar eros. "
         "Curabitur cursus molestie dolor eu ultricies. Suspendisse auctor massa et mi semper "
         "ultricies. Duis ac leo hendrerit, porta urna in, bibendum arcu. Maecenas auctor aliquet "
         "tortor, ut pellentesque justo iaculis vulputate. Pellentesque ut viverra nulla. Quisque "
         "placerat sed lacus quis rhoncus. Interdum et malesuada fames ac ante ipsum primis in "
         "faucibus.")


def correr(bits, mod, codigo, params, snr, rng):
    """Una transmision completa midiendo el tiempo de procesamiento."""
    t0 = time.perf_counter()
    r = app.cadena_tx_codif_rx(bits, mod, codigo, params, snr, rng)
    r["tiempo_s"] = time.perf_counter() - t0
    return r


# =====================================================================
# FIGURA 1: esquema de los codificadores (convolucional y turbo)
# =====================================================================
fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4))

# --- (a) codificador convolucional tail-biting, tasa 1/3 ---
ax = axes[0]
ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")
ax.set_title("(a) Codificador convolucional tail-biting LTE, tasa 1/3 (64 estados)",
             fontsize=11, weight="bold", loc="left")
# Registro de 6 memorias
for i in range(6):
    x = 2.3 + i * 1.3
    ax.add_patch(Rectangle((x, 3.6), 1.0, 0.9, fc="#EAF0FB", ec="#1C3A58", lw=1.4))
    ax.text(x + 0.5, 4.05, f"D{i+1}", ha="center", va="center", fontsize=10, weight="bold")
ax.annotate("", xy=(2.3, 4.05), xytext=(0.6, 4.05), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(0.55, 4.35, "entrada $c_k$", fontsize=10)
for i in range(5):
    x = 2.3 + i * 1.3 + 1.0
    ax.annotate("", xy=(x + 0.3, 4.05), xytext=(x, 4.05), arrowprops=dict(arrowstyle="->", lw=1.2))
# Taps de cada generador (octal -> binario: [entrada, D1..D6])
GEN = [("$d_k^{(0)}$  $G_0=133_8$", 0o133, 2.6), ("$d_k^{(1)}$  $G_1=171_8$", 0o171, 1.6),
       ("$d_k^{(2)}$  $G_2=165_8$", 0o165, 0.6)]
xs_tap = [1.45] + [2.3 + i * 1.3 + 0.5 for i in range(6)]     # x de entrada y de cada D
for nombre, g, y in GEN:
    taps = [(g >> (6 - j)) & 1 for j in range(7)]
    xs = [xs_tap[j] for j, t in enumerate(taps) if t]
    ax.plot([xs[0], 11.0], [y, y], color="#1C3A58", lw=1.2)
    for x in xs:
        ax.plot([x, x], [y, 3.6 if x > 2 else 4.05], color="#9FB2C8", lw=0.9, ls=":")
        ax.add_patch(Circle((x, y), 0.09, fc="#1E54E0", ec="none"))
    ax.annotate("", xy=(11.6, y), xytext=(11.0, y), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(11.65, y, nombre, fontsize=9.5, va="center")
ax.text(0.6, 2.9, "tail-biting: el registro se inicializa\ncon los últimos 6 bits del bloque",
        fontsize=8.5, style="italic", color="#444")

# --- (b) turbo codificador PCCC, tasa 1/3 ---
ax = axes[1]
ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")
ax.set_title("(b) Turbo codificador LTE (PCCC): 2 RSC de 8 estados + entrelazador QPP",
             fontsize=11, weight="bold", loc="left")
def caja(ax, x, y, w, h, texto, fc="#EAF0FB"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="#1C3A58", lw=1.4))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=9.5, weight="bold")
caja(ax, 4.4, 3.7, 3.4, 1.1, "RSC 1 (8 estados)\n$g_0=1+D^2+D^3$,  $g_1=1+D+D^3$")
caja(ax, 4.4, 0.6, 3.4, 1.1, "RSC 2 (8 estados)\n(mismo codificador)")
caja(ax, 1.9, 0.6, 1.6, 1.1, "QPP\n$\\pi(i)$", fc="#FFF3E0")
ax.annotate("", xy=(4.4, 4.25), xytext=(0.6, 4.25), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(0.55, 4.55, "entrada $c_k$ (bloque de $K$ bits)", fontsize=10)
ax.plot([1.3, 1.3], [4.25, 1.15], color="#1C3A58", lw=1.2)
ax.add_patch(Circle((1.3, 4.25), 0.07, fc="#1C3A58"))
ax.annotate("", xy=(1.9, 1.15), xytext=(1.3, 1.15), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(4.4, 1.15), xytext=(3.5, 1.15), arrowprops=dict(arrowstyle="->", lw=1.2))
# Salidas
ax.plot([1.3, 1.3], [4.25, 5.05], color="#1C3A58", lw=1.2)
ax.plot([1.3, 11.0], [5.05, 5.05], color="#1C3A58", lw=1.2)
ax.annotate("", xy=(11.6, 5.05), xytext=(11.0, 5.05), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(11.65, 5.05, "$S_k$ (sistemático)", fontsize=9.5, va="center")
ax.annotate("", xy=(11.6, 4.25), xytext=(7.8, 4.25), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(11.65, 4.25, "$P1_k$ (paridad 1)", fontsize=9.5, va="center")
ax.annotate("", xy=(11.6, 1.15), xytext=(7.8, 1.15), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(11.65, 1.15, "$P2_k$ (paridad 2)", fontsize=9.5, va="center")
ax.text(4.4, 2.6, "$\\pi(i) = (f_1\\,i + f_2\\,i^2)\\;\\mathrm{mod}\\;K$   (K=512: $f_1=31$, $f_2=64$)",
        fontsize=9.5, color="#444")
plt.tight_layout()
plt.savefig(f"{IMGDIR}/diagrama_codificadores.png", dpi=150, bbox_inches="tight")
plt.close()
print("FIG diagrama_codificadores.png lista", flush=True)

# =====================================================================
# DEMO DE TEXTO: lorem ipsum sin codificar / convolucional / turbo
# =====================================================================
MOD_TXT, SNR_TXT = "16-QAM", 7.0
bits_txt = app.texto_a_bits(LOREM)
params_txt = app.calcular_parametros_ofdm(BW, DF, CP, len(bits_txt), 4)
demo = {"snr_db": SNR_TXT, "modulacion": MOD_TXT, "n_bits": int(len(bits_txt)),
        "n_bloques": int(-(-len(bits_txt) // 512))}
rng = np.random.default_rng(42)
for codigo in CODIGOS:
    r = correr(bits_txt, MOD_TXT, codigo, params_txt, SNR_TXT, rng)
    demo[codigo] = {"ber": float(r["ber"]), "tiempo_ms": r["tiempo_s"] * 1e3,
                    "texto": app.bits_a_texto(r["bits_rx"])}
    print(f"TXT {codigo:14s} BER={r['ber']:.4f} t={r['tiempo_s']*1e3:.0f} ms", flush=True)
with open(os.path.join(IMGDIR, "..", "demo_texto.txt"), "w", encoding="utf-8") as f:
    for codigo in CODIGOS:
        f.write(f"===== {ETQ[codigo]} | BER={demo[codigo]['ber']:.4f} | "
                f"t={demo[codigo]['tiempo_ms']:.0f} ms =====\n")
        f.write(demo[codigo]["texto"][:400] + "\n\n")
resultados["demo_texto"] = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "texto"})
                            for k, v in demo.items()}

# =====================================================================
# FIGURA 2: recuperacion de Lena (sin codificar / convolucional / turbo)
# =====================================================================
MOD_IMG, SNR_IMG = "16-QAM", 7.0
lena = Image.open("lena.png").convert("RGB").resize((96, 96), Image.LANCZOS)
bits_img, modo, size = app.imagen_a_bits(lena)
params_img = app.calcular_parametros_ofdm(BW, DF, CP, len(bits_img), 4)
print(f"Lena 96x96 RGB -> {len(bits_img)} bits ({-(-len(bits_img)//512)} bloques)", flush=True)
fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.3))
axes[0].imshow(lena); axes[0].set_title("Original", fontsize=11, weight="bold"); axes[0].axis("off")
resultados["lena"] = {"snr_db": SNR_IMG, "modulacion": MOD_IMG, "n_bits": int(len(bits_img)),
                      "n_bloques": int(-(-len(bits_img) // 512))}
for ax, codigo in zip(axes[1:], CODIGOS):
    rng = np.random.default_rng(300)
    r = correr(bits_img, MOD_IMG, codigo, params_img, SNR_IMG, rng)
    ax.imshow(app.bits_a_imagen(r["bits_rx"], modo, size))
    ax.set_title(f"{ETQ[codigo]}\nBER = {r['ber']:.2e}  ({r['tiempo_s']:.2f} s)",
                 fontsize=9.5, weight="bold", color=COL[codigo])
    ax.axis("off")
    resultados["lena"][codigo] = {"ber": float(r["ber"]), "tiempo_s": r["tiempo_s"]}
    print(f"LENA {codigo:14s} BER={r['ber']:.2e} t={r['tiempo_s']:.2f} s", flush=True)
fig.suptitle(f"Transmisión de Lena (96×96) — {MOD_IMG}, SNR = {SNR_IMG:.0f} dB, canal Pedestrian A",
             fontsize=12, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{IMGDIR}/lena_codificacion.png", dpi=150, bbox_inches="tight")
plt.close()
print("FIG lena_codificacion.png lista", flush=True)

# =====================================================================
# FIGURAS 3 y 4: BER vs SNR y tiempo de procesamiento vs SNR (Monte Carlo
# transmitiendo Lena en cada corrida) para QPSK y 16-QAM
# =====================================================================
SNRS = list(range(0, 16))
N_SIM = 3
MODS = ["QPSK", "16-QAM"]
PISO = 1e-6
tcrit = float(t_student.ppf(0.975, df=N_SIM - 1))
curvas = {}       # (mod, codigo) -> dict con ber mu/lo/hi y tiempo mu
rng = np.random.default_rng(2026)
for mod in MODS:
    bps = app.MODULACIONES[mod]["bits"]
    params = app.calcular_parametros_ofdm(BW, DF, CP, len(bits_img), bps)
    for codigo in CODIGOS:
        mu, lo, hi, tmu = [], [], [], []
        t_ini = time.perf_counter()
        for snr in SNRS:
            bers, ts = [], []
            for _ in range(N_SIM):
                r = correr(bits_img, mod, codigo, params, snr, rng)
                bers.append(r["ber"]); ts.append(r["tiempo_s"])
            bers = np.array(bers)
            m, s = bers.mean(), bers.std(ddof=1)
            mar = tcrit * s / np.sqrt(N_SIM)
            mu.append(max(m, PISO)); lo.append(max(m - mar, PISO)); hi.append(max(m + mar, PISO))
            tmu.append(float(np.mean(ts)))
        curvas[(mod, codigo)] = {"ber": mu, "lo": lo, "hi": hi, "t": tmu}
        print(f"CURVA {mod} {codigo:14s} lista en {time.perf_counter()-t_ini:.0f} s "
              f"(BER@4dB={mu[4]:.2e}, t_medio={np.mean(tmu):.2f} s)", flush=True)

# --- FIGURA 3: BER vs SNR ---
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
for ax, mod in zip(axes, MODS):
    for codigo in CODIGOS:
        c = curvas[(mod, codigo)]
        ax.plot(SNRS, c["ber"], "-o", color=COL[codigo], ms=4, lw=1.8, label=ETQ[codigo])
        ax.fill_between(SNRS, c["lo"], c["hi"], color=COL[codigo], alpha=0.18)
    ax.set_yscale("log"); ax.set_ylim(PISO, 1); ax.set_xlim(0, 15)
    ax.set_xlabel("SNR (dB)"); ax.set_title(mod, fontsize=12, weight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower left")
axes[0].set_ylabel("BER (escala log)")
fig.suptitle("BER vs SNR — códigos de canal LTE sobre OFDM + Pedestrian A "
             f"(Monte Carlo, {N_SIM} corridas de {len(bits_img)//1000} kbits, IC 95 %)",
             fontsize=12, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{IMGDIR}/ber_snr.png", dpi=150, bbox_inches="tight")
plt.close()
print("FIG ber_snr.png lista", flush=True)

# --- FIGURA 4: tiempo de procesamiento vs SNR ---
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
for ax, mod in zip(axes, MODS):
    for codigo in CODIGOS:
        c = curvas[(mod, codigo)]
        ax.plot(SNRS, np.array(c["t"]) * 1e3, "-o", color=COL[codigo], ms=4, lw=1.8, label=ETQ[codigo])
    ax.set_xlim(0, 15); ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)"); ax.set_title(mod, fontsize=12, weight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="center right")
axes[0].set_ylabel("Tiempo de procesamiento por transmisión (ms)")
fig.suptitle(f"Tiempo de ejecución de la cadena vs SNR — transmisión de Lena "
             f"({len(bits_img)//1000} kbits, {-(-len(bits_img)//512)} bloques de K=512)",
             fontsize=12, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"{IMGDIR}/tiempo_snr.png", dpi=150, bbox_inches="tight")
plt.close()
print("FIG tiempo_snr.png lista", flush=True)

# --- Numeros clave para el informe ---
resumen = {"snrs": SNRS, "n_sim": N_SIM, "n_bits": int(len(bits_img)),
           "n_bloques": int(-(-len(bits_img) // 512))}
for mod in MODS:
    resumen[mod] = {}
    for codigo in CODIGOS:
        c = curvas[(mod, codigo)]
        t_med = float(np.mean(c["t"]))
        resumen[mod][codigo] = {
            "ber_0dB": c["ber"][0], "ber_4dB": c["ber"][4], "ber_8dB": c["ber"][8],
            "ber_12dB": c["ber"][12],
            "t_medio_s": t_med, "t_por_bloque_ms": t_med / resumen["n_bloques"] * 1e3,
        }
resultados["curvas"] = resumen
with open(os.path.join(IMGDIR, "..", "resultados.json"), "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("OK: figuras + resultados.json + demo_texto.txt generados", flush=True)
