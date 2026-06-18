# Informe Práctica 2 — SC-FDM (carpeta `informe-scfdm/`)

Carpeta **independiente** de la Práctica 1 (OFDM, en `informe/`) para no mezclar las prácticas.
Para compilar en Overleaf: subir `informe.tex`, `referencias.bib` y la carpeta `img/`.

Todas las capturas ya están generadas en `img/` (tomadas del simulador desplegado en
`ofdm.tesisucuenca.com`, pestaña SC-FDM). Configuración usada: BW 5 MHz, Δf 15 kHz, CP normal.

| Archivo en `img/`            | Figura (label)   | Qué muestra | Dato medido |
|------------------------------|------------------|-------------|-------------|
| `qr_sitio.png`               | `fig:qrsitio`    | QR del sitio desplegado | ofdm.tesisucuenca.com |
| `qr_repo.png`                | `fig:qrrepo`     | QR del repositorio | github.com/thyron001/ofdm |
| `diagrama_bloques.png`       | `fig:diagrama`   | Diagrama de bloques de la cadena SC-FDM (DFT/IDFT resaltadas) | — |
| `interfaz_scfdm.png`         | `fig:interfaz`   | Mapa de cobertura SC-FDM con 2 zonas | 16-QAM 0–600 m, QPSK 600–1500 m |
| `params_scfdm.png`           | `fig:params`     | Parámetros calculados SC-FDM | N_SC 300, N_FFT 512, M 275, mapeo localizado |
| `sim_16qam_scfdm.png`        | `fig:sim16`      | TX/RX + BER con 16-QAM | SNR 25 dB → BER 5,2×10⁻⁴ |
| `sim_qpsk_scfdm.png`         | `fig:simqpsk`    | TX/RX + BER con QPSK | SNR 10 dB, 1039 m → BER 5,3×10⁻³ |
| `constelacion_scfdm.png`     | `fig:const`      | Constelación RX 16-QAM (ZF + IDFT-despread) | nubes en los 16 puntos |
| `mapa_sc_scfdm.png`          | `fig:mapasc`     | Bloque contiguo de datos (azul) + guarda (gris) | 275 datos + 25 guarda |
| `forma_onda.png`             | `fig:onda`       | Envolvente \|x(t)\| OFDM vs SC-FDM | PAPR 9,93 dB (OFDM) vs 5,97 dB (SC-FDM) |
| `montecarlo_comparacion.png` | `fig:montecarlo` | BER vs SNR, 5 curvas (OFDM continua + SC-FDM discontinua) | SC-FDM ≈ OFDM |
| `papr_comparacion.png`       | `fig:papr`       | CCDF del PAPR, 5 curvas combinadas | SC-FDM desplazado ~3–4 dB a la izquierda |
