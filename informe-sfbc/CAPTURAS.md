# Informe Práctica 4 — Diversidad en TX / SFBC (carpeta `informe-sfbc/`)

Carpeta **independiente** de las demás prácticas para no mezclarlas.
Para compilar en Overleaf: subir `informe.tex`, `referencias.bib` y la carpeta `img/`.

## Figuras

A diferencia del informe de SC-FDM (capturas manuales del simulador desplegado), **las figuras de
datos de este informe se generan automáticamente** con `generar_figuras.py`, a partir de corridas
reales del propio simulador (`app.py`). Todas están ya en `img/` y el informe compila tal cual.
Configuración base: BW 5 MHz, Δf 15 kHz, CP normal, modulación 16-QAM.

| Archivo en `img/`        | Figura (label)   | Qué muestra | Origen |
|--------------------------|------------------|-------------|--------|
| `diagrama_sfbc.png`      | `fig:diagrama`   | Diagrama de bloques de la cadena SFBC (codif./decod. resaltados) | matplotlib |
| `params_sfbc.png`        | `fig:params`     | Tabla de parámetros (N_SC 300, N_FFT 512, 150 parejas Alamouti) | matplotlib |
| `sim_2x1_sfbc.png`       | `fig:sim2x1`     | Imagen original vs recuperada, SFBC 2×1, 16-QAM, 12 dB | datos reales (lena 256×256) |
| `sim_2x2_sfbc.png`       | `fig:sim2x2`     | Imagen original vs recuperada, SFBC 2×2, 16-QAM, 12 dB | datos reales (lena 256×256) |
| `constelacion_sfbc.png`  | `fig:const`      | Constelación RX 1×1 (dispersa) vs 2×2 (compacta), 15 dB | datos reales |
| `montecarlo_sfbc.png`    | `fig:montecarlo` | BER vs SNR, 3 curvas (1×1/2×1/2×2) con IC 95% | Monte Carlo, 8 corridas |
| `qr_sitio.png`           | `fig:qrsitio`    | QR del sitio | reutilizado de la Práctica 2 |
| `qr_repo.png`            | `fig:qrrepo`     | QR del repositorio | reutilizado de la Práctica 2 |

## Valores medidos (coinciden con el informe)

Transmisión de imagen (lena 256×256, 16-QAM, SNR 12 dB, una corrida) — Tabla I:

| Configuración | BER |
|---------------|-----|
| 1×1 (sin diversidad) | 1,9×10⁻² |
| SFBC 2×1 | 1,1×10⁻² |
| SFBC 2×2 | 1,4×10⁻³ |

Monte Carlo BER vs SNR (16-QAM, 8 corridas/punto) — Tabla II:

| SNR (dB) | 1×1     | SFBC 2×1 | SFBC 2×2 |
|----------|---------|----------|----------|
| 6        | 1,1×10⁻¹| 9,6×10⁻² | 4,8×10⁻² |
| 9        | 5,2×10⁻²| 4,3×10⁻² | 1,3×10⁻² |
| 12       | 2,2×10⁻²| 1,3×10⁻² | 1,8×10⁻³ |
| 15       | 7,1×10⁻³| 1,5×10⁻³ | 5,3×10⁻⁵ |
| 18       | 2,6×10⁻³| 2,8×10⁻⁴ | <10⁻⁵    |

## Regenerar las figuras

```bash
cd informe-sfbc
python generar_figuras.py     # requiere numpy, scipy, matplotlib, Pillow y ../lena.png
```

El script importa `app.py` (un nivel arriba) y usa `../lena.png` para las figuras de imagen.
El BER exacto varía levemente en cada corrida porque el canal Rayleigh es aleatorio; si se
regeneran las figuras, ajustar los valores de las Tablas I y II del `.tex` si difieren de forma
notable. Penalización medida de SFBC 2×1 frente a MRC 1×2 (misma diversidad orden 2): ≈ 2–3 dB.
