// === Simulador OFDM-LTE — Frontend ===

const $ = (id) => document.getElementById(id);

// ----- Estado de la UI -----
const estado = {
  imagenSubida: false,
  nBits: 0,
  modulacion: '64-QAM',
  distancia: 0,
  uePos: { x: 250, y: 250 },
  arrastrando: false,
  graficoMC: null,
  graficoConst: null,
};

// ----- Combinaciones LTE válidas -----
const COMBINACIONES_NO_VALIDAS = new Set([
  '1.4_7.5', '3_7.5',
]);

// ===========================================================
// === SIDEBAR: cambios en BW/Δf/CP                         ===
// ===========================================================
function actualizarOpcionesCP() {
  const df = $('df').value;
  const cp = $('cp');
  cp.innerHTML = '';
  if (df === '15') {
    cp.innerHTML = '<option value="normal">Normal (4.7 µs)</option>' +
                   '<option value="extendido">Extendido (16.67 µs)</option>';
  } else {
    cp.innerHTML = '<option value="extendido">Extendido (33.33 µs)</option>';
  }
}

function validarCombinacion() {
  const bw = $('bw').value;
  const df = $('df').value;
  const aviso = $('aviso-combinacion');
  const k = `${bw}_${df}`;
  if (COMBINACIONES_NO_VALIDAS.has(k)) {
    aviso.style.display = 'block';
    return false;
  }
  aviso.style.display = 'none';
  return true;
}

async function actualizarParametros() {
  if (!validarCombinacion()) {
    $('kv-params').innerHTML = '<div class="kv"><div class="kv-label">Combinación inválida</div></div>';
    return;
  }
  const payload = {
    bw_mhz: parseFloat($('bw').value),
    delta_f_khz: parseFloat($('df').value),
    tipo_cp: $('cp').value,
    modulacion: estado.modulacion,
    n_bits: estado.nBits,
  };
  try {
    const r = await fetch('/calcular_parametros', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const j = await r.json();
    if (j.error) {
      $('kv-params').innerHTML = `<div class="kv"><div class="kv-label">Error</div><div class="kv-valor" style="font-size:13px">${j.error}</div></div>`;
      return;
    }
    renderKV('kv-params', j);
  } catch (e) {
    console.error(e);
  }
}

function renderKV(contenedorId, p, extras = {}) {
  const items = [
    ['Subportadoras útiles (N_SC)',         p.n_sc],
    ['Tamaño de la FFT (N_FFT)',            p.n_fft],
    ['Frecuencia de muestreo (fs)',         (p.fs / 1e6).toFixed(3) + ' MHz'],
    ['Muestras de prefijo cíclico (N_CP)',  p.n_cp],
    ['Subportadoras piloto',                p.n_pilotos],
    ['Subportadoras de datos',              p.n_datos],
    ['Bits totales de la imagen',           (estado.nBits || 0).toLocaleString()],
    ['Símbolos QAM totales',                p.n_simbolos_qam.toLocaleString()],
    ['Símbolos OFDM transmitidos',          p.n_simbolos_ofdm.toLocaleString()],
    ['Duración de cada símbolo OFDM',       p.duracion_simbolo_us.toFixed(2) + ' µs'],
    ['Duración del prefijo cíclico',        p.duracion_cp_us.toFixed(2) + ' µs'],
    ['Modulación aplicada',                 estado.modulacion || '—'],
  ];
  if (extras.tiempo_aire_s != null) {
    items.push(['Tiempo de transmisión (aire)', formatearTiempo(extras.tiempo_aire_s)]);
  }
  $(contenedorId).innerHTML = items.map(([l, v]) =>
    `<div class="kv"><div class="kv-label">${l}</div><div class="kv-valor">${v}</div></div>`
  ).join('');
}

function formatearTiempo(s) {
  if (s < 1e-3) return (s * 1e6).toFixed(2) + ' µs';
  if (s < 1) return (s * 1e3).toFixed(3) + ' ms';
  return s.toFixed(3) + ' s';
}
function formatearTasa(bps) {
  if (bps >= 1e6) return (bps / 1e6).toFixed(2) + ' Mbps';
  if (bps >= 1e3) return (bps / 1e3).toFixed(2) + ' kbps';
  return bps.toFixed(0) + ' bps';
}

// ===========================================================
// === Slider SNR sincronizado                              ===
// ===========================================================
function sincSNR() {
  $('snr').addEventListener('input', () => { $('snr-num').value = $('snr').value; });
  $('snr-num').addEventListener('input', () => {
    let v = parseInt($('snr-num').value || '0');
    if (v < 0) v = 0; if (v > 100) v = 100;
    $('snr').value = v;
  });
}

// ===========================================================
// === Carga de imagen                                      ===
// ===========================================================
async function subirImagen(e) {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('imagen', file);
  $('info-imagen').textContent = 'Subiendo...';
  const r = await fetch('/subir_imagen', { method: 'POST', body: fd });
  const j = await r.json();
  if (j.error) {
    $('info-imagen').textContent = 'Error: ' + j.error;
    $('info-imagen').className = 'estado err';
    return;
  }
  estado.imagenSubida = true;
  estado.nBits = j.n_bits;
  $('preview-imagen').src = j.preview_b64;
  $('preview-imagen').style.display = 'block';
  $('info-imagen').className = 'estado ok';
  $('info-imagen').textContent = `${j.ancho}×${j.alto}, ${j.canales} canal(es), ${j.n_bits.toLocaleString()} bits ✓`;
  actualizarParametros();
}

// ===========================================================
// === Canvas de cobertura                                  ===
// ===========================================================
const CANVAS_TAM = 500;
const CENTRO = { x: CANVAS_TAM / 2, y: CANVAS_TAM / 2 };
const RADIO_MAX_M = 1500;            // metros
const RADIO_MAX_PX = 230;            // px (deja margen)
const M_POR_PX = RADIO_MAX_M / RADIO_MAX_PX;

const ZONAS = [
  { rMin: 0,    rMax: 300,  mod: '64-QAM', color: 'rgba(24,163,75,0.18)', borde: '#18A34B' },
  { rMin: 300,  rMax: 800,  mod: '16-QAM', color: 'rgba(30,84,224,0.14)', borde: '#1E54E0' },
  { rMin: 800,  rMax: 1500, mod: 'QPSK',   color: 'rgba(229,138,26,0.16)', borde: '#E58A1A' },
];

function dibujarCobertura() {
  const cv = $('canvas-cobertura');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);

  // Fondo de cuadrícula sutil
  ctx.strokeStyle = '#E4EAF0';
  ctx.lineWidth = 1;
  for (let i = 50; i < CANVAS_TAM; i += 50) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, CANVAS_TAM); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(CANVAS_TAM, i); ctx.stroke();
  }

  // Anillos (de afuera hacia adentro para que el verde quede encima)
  [...ZONAS].reverse().forEach((z) => {
    const rPxMax = z.rMax / M_POR_PX;
    ctx.beginPath();
    ctx.arc(CENTRO.x, CENTRO.y, rPxMax, 0, Math.PI * 2);
    ctx.fillStyle = z.color;
    ctx.fill();
    ctx.strokeStyle = z.borde;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  });

  // Etiquetas
  ctx.fillStyle = '#0B2540';
  ctx.font = 'bold 12px Nunito';
  ctx.textAlign = 'center';
  ZONAS.forEach((z) => {
    const rMidPx = ((z.rMin + z.rMax) / 2) / M_POR_PX;
    ctx.fillText(`${z.mod} (${z.rMin}–${z.rMax} m)`, CENTRO.x, CENTRO.y - rMidPx + 4);
  });

  // Antena (triángulo)
  ctx.fillStyle = '#1C3A58';
  ctx.beginPath();
  ctx.moveTo(CENTRO.x, CENTRO.y - 14);
  ctx.lineTo(CENTRO.x - 10, CENTRO.y + 8);
  ctx.lineTo(CENTRO.x + 10, CENTRO.y + 8);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#1E54E0';
  ctx.beginPath();
  ctx.arc(CENTRO.x, CENTRO.y, 4, 0, Math.PI * 2);
  ctx.fill();

  // UE (móvil)
  const fuera = estado.distancia > RADIO_MAX_M;
  ctx.fillStyle = fuera ? '#D82A2A' : '#1642B4';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(estado.uePos.x, estado.uePos.y, 10, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Escala
  ctx.fillStyle = '#5B6B7E';
  ctx.font = 'bold 10px Nunito';
  ctx.textAlign = 'left';
  ctx.fillText(`Escala: ${Math.round(M_POR_PX)} m/px`, 10, CANVAS_TAM - 12);
}

function calcDistancia() {
  const dx = estado.uePos.x - CENTRO.x;
  const dy = estado.uePos.y - CENTRO.y;
  return Math.round(Math.sqrt(dx * dx + dy * dy) * M_POR_PX);
}

function modulacionPorDistancia(d) {
  if (d > RADIO_MAX_M) return null;
  const z = ZONAS.find((z) => d >= z.rMin && d < z.rMax);
  return z ? z.mod : ZONAS[ZONAS.length - 1].mod;
}

function actualizarUE() {
  estado.distancia = calcDistancia();
  const mod = modulacionPorDistancia(estado.distancia);
  $('distancia-ue').textContent = estado.distancia;
  const badge = $('badge-modulacion');
  if (mod === null) {
    badge.textContent = 'Fuera de cobertura';
    badge.className = 'badge badge-rojo';
    estado.modulacion = null;
    $('btn-simular').disabled = true;
  } else {
    estado.modulacion = mod;
    badge.textContent = mod;
    badge.className = 'badge ' + (mod === '64-QAM' ? 'badge-verde' :
                                  mod === '16-QAM' ? 'badge-azul' : 'badge-naranja');
    $('btn-simular').disabled = !estado.imagenSubida;
  }
  dibujarCobertura();
  actualizarParametros();
}

function bindCobertura() {
  const cv = $('canvas-cobertura');
  const posMouse = (ev) => {
    const r = cv.getBoundingClientRect();
    return {
      x: (ev.clientX - r.left) * (cv.width / r.width),
      y: (ev.clientY - r.top) * (cv.height / r.height),
    };
  };
  cv.addEventListener('mousedown', (ev) => {
    const p = posMouse(ev);
    const dx = p.x - estado.uePos.x, dy = p.y - estado.uePos.y;
    if (Math.sqrt(dx * dx + dy * dy) < 20) estado.arrastrando = true;
    else { estado.uePos = p; actualizarUE(); }
  });
  cv.addEventListener('mousemove', (ev) => {
    if (!estado.arrastrando) return;
    estado.uePos = posMouse(ev);
    actualizarUE();
  });
  window.addEventListener('mouseup', () => { estado.arrastrando = false; });
}

// ===========================================================
// === Simulación normal                                    ===
// ===========================================================
async function ejecutarSimulacion() {
  if (!estado.imagenSubida) { alert('Sube una imagen primero'); return; }
  if (!estado.modulacion) { alert('UE fuera de cobertura'); return; }

  $('btn-simular').disabled = true;
  $('btn-simular').innerHTML = '<span class="spinner"></span> Simulando...';
  $('estado-sim').textContent = 'Ejecutando cadena TX → Canal → AWGN → RX...';
  $('estado-sim').className = 'estado';

  const payload = {
    bw_mhz: parseFloat($('bw').value),
    delta_f_khz: parseFloat($('df').value),
    tipo_cp: $('cp').value,
    snr_db: parseFloat($('snr').value),
    modulacion: estado.modulacion,
  };
  try {
    const r = await fetch('/simular', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    pintarResultados(j);
    $('estado-sim').textContent = 'Simulación completada ✓';
    $('estado-sim').className = 'estado ok';
  } catch (e) {
    $('estado-sim').textContent = 'Error: ' + e.message;
    $('estado-sim').className = 'estado err';
  } finally {
    $('btn-simular').disabled = false;
    $('btn-simular').textContent = 'Simular Transmisión';
  }
}

function pintarResultados(j) {
  $('zona-resultados').style.display = 'block';
  $('img-tx').src = j.imagen_original_b64;
  $('img-rx').src = j.imagen_recuperada_b64;
  $('ber-valor').textContent = (j.ber * 100).toFixed(3) + ' %';
  $('ber-detalle').textContent = `${j.bits_erroneos.toLocaleString()} / ${j.bits_transmitidos.toLocaleString()} bits — ${j.n_simbolos_ofdm} símbolos OFDM — SNR ${j.snr_db} dB`;

  dibujarConstelacion(j.constelacion_tx, j.constelacion_rx, j.modulacion);
  dibujarMapaSC(j.mapa_subportadoras);
  // Refrescar la tarjeta superior con todos los parámetros + tiempo
  renderKV('kv-params', j.parametros, {
    tiempo_aire_s: j.tiempo_aire_s,
  });
  $('titulo-constelacion').textContent = `Constelación RX — ${j.modulacion}`;
  $('resumen-sc').textContent = `Datos: ${j.mapa_subportadoras.datos} · Pilotos: ${j.mapa_subportadoras.pilotos} · Total: ${j.mapa_subportadoras.total}`;
}

function dibujarConstelacion(tx, rx, mod) {
  const cv = $('constelacion-canvas');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);

  // Escala
  const lim = mod === 'QPSK' ? 1.2 : (mod === '16-QAM' ? 1.4 : 1.6);
  const sx = (x) => W / 2 + (x / lim) * (W / 2 - 10);
  const sy = (y) => H / 2 - (y / lim) * (H / 2 - 10);

  // Ejes
  ctx.strokeStyle = '#E4EAF0';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H);
  ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

  // Puntos RX (verde semitransparente)
  ctx.fillStyle = 'rgba(24,163,75,0.45)';
  for (let i = 0; i < rx.real.length; i++) {
    ctx.beginPath();
    ctx.arc(sx(rx.real[i]), sy(rx.imag[i]), 2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Puntos ideales TX (cruces rojas) — dedup primero
  const dedupe = new Set();
  const claves = [];
  for (let i = 0; i < tx.real.length; i++) {
    const k = `${tx.real[i].toFixed(2)}|${tx.imag[i].toFixed(2)}`;
    if (!dedupe.has(k)) { dedupe.add(k); claves.push([tx.real[i], tx.imag[i]]); }
  }
  ctx.strokeStyle = '#D82A2A';
  ctx.lineWidth = 2;
  claves.forEach(([r, i]) => {
    const x = sx(r), y = sy(i);
    ctx.beginPath();
    ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5);
    ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5);
    ctx.stroke();
  });
}

function dibujarMapaSC(mapa) {
  const cv = $('mapa-sc-canvas');
  const ctx = cv.getContext('2d');
  // Resolución más alta
  const W = cv.clientWidth || 800;
  cv.width = W; cv.height = 120;
  ctx.clearRect(0, 0, W, 120);
  const total = mapa.total;
  const ancho = W / total;
  const pilotosSet = new Set(mapa.indices_pilotos);
  for (let k = 0; k < total; k++) {
    let color;
    if (pilotosSet.has(k)) color = '#D82A2A';
    else color = '#1E54E0';
    ctx.fillStyle = color;
    ctx.fillRect(k * ancho, 10, Math.max(ancho - 0.5, 1), 90);
  }
  // Leyenda
  ctx.fillStyle = '#5B6B7E';
  ctx.font = 'bold 11px Nunito';
  ctx.fillText('■ Datos', 4, 115);
  ctx.fillStyle = '#1E54E0';
  ctx.fillRect(4, 107, 10, 8);
  ctx.fillStyle = '#5B6B7E';
  ctx.fillText('Datos', 18, 115);
  ctx.fillStyle = '#D82A2A';
  ctx.fillRect(80, 107, 10, 8);
  ctx.fillStyle = '#5B6B7E';
  ctx.fillText('Pilotos (cada 12)', 94, 115);
}

// ===========================================================
// === Monte Carlo                                          ===
// ===========================================================
async function ejecutarMontecarlo() {
  $('btn-mc').disabled = true;
  $('btn-mc').innerHTML = '<span class="spinner"></span> Ejecutando Monte Carlo...';
  $('estado-sim').textContent = 'Monte Carlo: 3 modulaciones × 11 SNR × 10 corridas...';
  $('estado-sim').className = 'estado';

  const payload = {
    bw_mhz: parseFloat($('bw').value),
    delta_f_khz: parseFloat($('df').value),
    tipo_cp: $('cp').value,
  };
  try {
    const r = await fetch('/montecarlo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    pintarMC(j);
    $('estado-sim').textContent = 'Monte Carlo completado ✓';
    $('estado-sim').className = 'estado ok';
  } catch (e) {
    $('estado-sim').textContent = 'Error: ' + e.message;
    $('estado-sim').className = 'estado err';
  } finally {
    $('btn-mc').disabled = false;
    $('btn-mc').textContent = 'Simulación Monte Carlo';
  }
}

// Plugin para dibujar barras de error en cada punto
const PLUGIN_ERROR_BARS = {
  id: 'errorBars',
  afterDatasetsDraw(chart) {
    const { ctx, scales: { x, y } } = chart;
    chart.data.datasets.forEach((ds, idx) => {
      if (!ds.errorBars) return;
      const meta = chart.getDatasetMeta(idx);
      if (meta.hidden) return;
      ctx.save();
      ctx.strokeStyle = ds.borderColor;
      ctx.lineWidth = 1.8;
      ds.data.forEach((pt, i) => {
        const eb = ds.errorBars[i];
        if (!eb) return;
        const px = x.getPixelForValue(pt.x);
        const pyMin = y.getPixelForValue(Math.max(eb.lo, y.min));
        const pyMax = y.getPixelForValue(Math.max(eb.hi, y.min));
        const w = 6;
        ctx.beginPath();
        ctx.moveTo(px, pyMin); ctx.lineTo(px, pyMax);
        ctx.moveTo(px - w, pyMin); ctx.lineTo(px + w, pyMin);
        ctx.moveTo(px - w, pyMax); ctx.lineTo(px + w, pyMax);
        ctx.stroke();
      });
      ctx.restore();
    });
  },
};

function pintarMC(j) {
  $('modal-mc').style.display = 'flex';
  const ctx = $('montecarlo-canvas').getContext('2d');
  if (estado.graficoMC) estado.graficoMC.destroy();
  if (estado.graficoPAPR) estado.graficoPAPR.destroy();

  const PISO = 1e-5;
  const colores = {
    'QPSK':   '#18A34B',
    '16-QAM': '#1E54E0',
    '64-QAM': '#D82A2A',
  };

  const datasets = [];
  for (const mod of ['QPSK', '16-QAM', '64-QAM']) {
    const r = j.resultados[mod];
    datasets.push({
      label: mod,
      data: r.ber_promedio.map((v, i) => ({ x: j.snr_valores[i], y: Math.max(v, PISO) })),
      errorBars: r.ber_promedio.map((_, i) => ({
        lo: Math.max(r.ic_inferior[i], PISO),
        hi: Math.max(r.ic_superior[i], PISO),
      })),
      borderColor: colores[mod],
      backgroundColor: colores[mod],
      pointRadius: 5,
      pointBackgroundColor: colores[mod],
      tension: 0.15,
      fill: false,
    });
  }

  estado.graficoMC = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'SNR (dB)', font: { weight: 700 } },
          ticks: { stepSize: 1 },
          min: -0.5, max: 15.5,
        },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'BER (escala logarítmica)', font: { weight: 700 } },
          min: PISO,
          max: 1,
        },
      },
      plugins: {
        legend: { position: 'top', labels: { font: { weight: 700 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const i = ctx.dataIndex;
              const ds = ctx.dataset;
              const eb = ds.errorBars[i];
              return `${ds.label}: BER=${ctx.parsed.y.toExponential(2)}  IC95% [${eb.lo.toExponential(2)}, ${eb.hi.toExponential(2)}]`;
            },
          },
        },
      },
    },
    plugins: [PLUGIN_ERROR_BARS],
  });

  // === CCDF del PAPR ===
  if (j.papr_ccdf) {
    const ctxP = $('papr-canvas').getContext('2d');
    const x = j.papr_ccdf.x_db;
    const dsP = ['QPSK', '16-QAM', '64-QAM'].map((mod) => ({
      label: mod,
      data: j.papr_ccdf[mod].map((v, i) => ({ x: x[i], y: Math.max(v, 1e-4) })),
      borderColor: colores[mod],
      backgroundColor: colores[mod],
      pointRadius: 0,
      borderWidth: 2,
      tension: 0.2,
      fill: false,
    }));
    estado.graficoPAPR = new Chart(ctxP, {
      type: 'line',
      data: { datasets: dsP },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'PAPR_0 (dB)', font: { weight: 700 } },
            min: 0, max: Math.max(...x),
          },
          y: {
            type: 'logarithmic',
            title: { display: true, text: 'Pr{PAPR > PAPR_0}', font: { weight: 700 } },
            min: 1e-3, max: 1,
          },
        },
        plugins: {
          legend: { position: 'top', labels: { font: { weight: 700 } } },
          title: {
            display: true,
            text: `CCDF del PAPR (${j.papr_ccdf.n_simbolos} símbolos OFDM por modulación)`,
          },
        },
      },
    });
  }
}

function descargarCanvas(canvasId, prefijo) {
  const src = $(canvasId);
  const tmp = document.createElement('canvas');
  tmp.width = src.width; tmp.height = src.height;
  const tctx = tmp.getContext('2d');
  tctx.fillStyle = '#FFFFFF';
  tctx.fillRect(0, 0, tmp.width, tmp.height);
  tctx.drawImage(src, 0, 0);
  const link = document.createElement('a');
  link.download = `${prefijo}_${Date.now()}.png`;
  link.href = tmp.toDataURL('image/png');
  link.click();
}

function descargarMC() {
  if (estado.graficoMC) descargarCanvas('montecarlo-canvas', 'montecarlo_ber_snr');
  if (estado.graficoPAPR) descargarCanvas('papr-canvas', 'papr_ccdf');
}

// ===========================================================
// === Inicialización                                       ===
// ===========================================================
window.addEventListener('DOMContentLoaded', () => {
  sincSNR();
  bindCobertura();
  $('df').addEventListener('change', () => { actualizarOpcionesCP(); actualizarParametros(); validarCombinacion(); });
  $('bw').addEventListener('change', () => { actualizarParametros(); validarCombinacion(); });
  $('cp').addEventListener('change', actualizarParametros);
  $('archivo-imagen').addEventListener('change', subirImagen);
  $('btn-simular').addEventListener('click', ejecutarSimulacion);
  $('btn-mc').addEventListener('click', ejecutarMontecarlo);

  // Lightbox
  ['img-tx', 'img-rx'].forEach((id) => {
    $(id).addEventListener('click', () => {
      const lb = $('lightbox');
      $('lightbox-img').src = $(id).src;
      lb.style.display = 'flex';
    });
  });
  $('lightbox').addEventListener('click', () => { $('lightbox').style.display = 'none'; });

  // Modal Monte Carlo
  $('btn-cerrar-mc').addEventListener('click', () => { $('modal-mc').style.display = 'none'; });
  $('btn-descargar-mc').addEventListener('click', descargarMC);
  $('modal-mc').addEventListener('click', (ev) => {
    if (ev.target.id === 'modal-mc') $('modal-mc').style.display = 'none';
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') {
      $('lightbox').style.display = 'none';
      $('modal-mc').style.display = 'none';
    }
  });

  actualizarUE();
  actualizarParametros();
  $('btn-simular').disabled = true;
});
