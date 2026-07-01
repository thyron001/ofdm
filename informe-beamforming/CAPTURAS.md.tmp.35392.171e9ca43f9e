# Figuras del informe — Beamforming (Práctica 5)

Todas las figuras de `img/` se generan automáticamente con datos reales del simulador
(no son capturas manuales). Para regenerarlas:

```bash
cd informe-beamforming
python generar_figuras.py
```

El script importa `app.py` (la cadena del sitio) y produce:

| Archivo | Contenido | Fuente |
|---|---|---|
| `montecarlo_bf.png`      | BER vs SNR — órdenes 1×1 / 2×1 / 4×1 (8 corridas, IC 95%) | `cadena_tx_beamforming_rx` |
| `montecarlo_overlay.png` | BER vs SNR — BF 2×1 vs SFBC 2×1 vs MRC 1×2 (mismo piso de ruido) | las 3 cadenas, `ruido_piso_fijo=True` |
| `constelacion_bf.png`    | Constelación 16-QAM RX a 15 dB: 1×1 vs BF 4×1 | `cadena_tx_beamforming_rx` (constelaciones) |
| `sim_2x1_bf.png`         | Imagen TX/RX, BF 2×1, 16-QAM, 12 dB | `cadena_tx_beamforming_rx` + `lena.png` |
| `sim_4x1_bf.png`         | Imagen TX/RX, BF 4×1, 16-QAM, 12 dB | `cadena_tx_beamforming_rx` + `lena.png` |
| `diagrama_bf.png`        | Diagrama de bloques de la cadena (precoder/detector resaltados) | matplotlib |
| `params_bf.png`          | Tabla de parámetros calculados | matplotlib |
| `qr_sitio.png`           | QR al simulador desplegado | copiado de `informe-sfbc/` |
| `qr_repo.png`            | QR al repositorio del proyecto | copiado de `informe-sfbc/` |

Compilación (local o en Overleaf): `pdflatex` → `bibtex` → `pdflatex` ×2.
