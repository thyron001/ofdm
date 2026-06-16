# Capturas para el informe (carpeta `img/`)

Guarda cada captura del chat con EXACTAMENTE este nombre dentro de `informe/img/`.
Las dos primeras (QR) ya están generadas.

| Archivo en `img/`        | Figura (label)     | Qué muestra | Dato medido |
|--------------------------|--------------------|-------------|-------------|
| `qr_sitio.png` ✅ ya está | `fig:qrsitio`      | QR del sitio desplegado | ofdm.tesisucuenca.com |
| `qr_repo.png`  ✅ ya está | `fig:qrrepo`       | QR del repositorio | github.com/thyron001/ofdm |
| `interfaz.png`           | `fig:interfaz`     | Interfaz completa: config + mapa de cobertura | BW 5 MHz, Δf 15 kHz, CP normal |
| `params_calculados.png`  | `fig:params`       | Tarjeta "Parámetros Calculados" | N_SC 300, N_FFT 512, fs 7,680 MHz, N_CP 36 |
| `sim_64qam.png`          | `fig:sim64`        | TX/RX + BER con 64-QAM | SNR 30 dB → BER 9,8×10⁻⁴ |
| `sim_16qam.png`          | `fig:sim16`        | TX/RX + BER con 16-QAM | SNR 15 dB → BER 8,5×10⁻³ |
| `sim_qpsk.png`           | `fig:simqpsk`      | TX/RX + BER con QPSK | SNR 6 dB → BER 1,8×10⁻² |
| `constelacion.png`       | `fig:const`        | Constelación RX 16-QAM (ZF) | nubes alrededor de los 16 puntos |
| `mapa_subportadoras.png` | `fig:mapasc`       | Franjas datos (azul) / piloto (rojo) | 275 datos + 25 pilotos = 300 |
| `montecarlo.png`         | `fig:montecarlo`   | BER vs SNR (0–15 dB), IC 95% t-Student | orden QPSK < 16-QAM < 64-QAM |
| `papr.png`               | `fig:papr`         | CCDF del PAPR (3 modulaciones) | curvas casi coincidentes, codo ~7 dB |

Compilar en Overleaf: subir `informe.tex`, `referencias.bib` y la carpeta `img/`.
