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

function renderKV(contenedorId, p) {
  const items = [
    ['N_SC',                p.n_sc],
    ['N_FFT',               p.n_fft],
    ['fs (MHz)',            (p.fs / 1e6).toFixed(3)],
    ['N_CP',                p.n_cp],
    ['Pilotos',             p.n_pilotos],
    ['Datos',               p.n_datos],
    ['Bits imagen',         estado.nBits || '—'],
    ['Símbolos QAM',        p.n_simbolos_qam],
    ['Símbolos OFDM',       p.n_simbolos_ofdm],
    ['Dur. símbolo (µs)',   p.duracion_simbolo_us.toFixed(2)],
    ['Dur. CP (µs)',        p.duracion_cp_us.toFixed(2)],
    ['Modulación',          estado.modulacion],
  ];
  $(contenedorId).innerHTML = items.map(([l, v]) =>
    `<div class="kv"><div class="kv-label">${l}</div><div class="kv-valor">${v}</div></div>`
  ).join('');
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
  renderKV('kv-params-resumen', j.parametros);
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

function pintarMC(j) {
  $('tarjeta-mc').style.display = 'block';
  const ctx = $('montecarlo-canvas').getContext('2d');
  if (estado.graficoMC) estado.graficoMC.destroy();

  const colores = {
    'QPSK':   { c: '#18A34B', tint: 'rgba(24,163,75,0.2)' },
    '16-QAM': { c: '#1E54E0', tint: 'rgba(30,84,224,0.2)' },
    '64-QAM': { c: '#D82A2A', tint: 'rgba(216,42,42,0.2)' },
  };

  const datasets = [];
  for (const mod of ['QPSK', '16-QAM', '64-QAM']) {
    const r = j.resultados[mod];
    datasets.push({
      label: mod,
      data: r.ber_promedio.map((v, i) => ({ x: j.snr_valores[i], y: Math.max(v, 1e-6) })),
      borderColor: colores[mod].c,
      backgroundColor: colores[mod].tint,
      pointRadius: 4,
      tension: 0.2,
      fill: false,
    });
    // Banda IC
    datasets.push({
      label: `${mod} IC95`,
      data: r.ic_superior.map((v, i) => ({ x: j.snr_valores[i], y: Math.max(v, 1e-6) })),
      borderColor: colores[mod].tint,
      backgroundColor: colores[mod].tint,
      pointRadius: 0,
      borderDash: [4, 4],
      fill: '+1',
    });
    datasets.push({
      label: `${mod} IC95-`,
      data: r.ic_inferior.map((v, i) => ({ x: j.snr_valores[i], y: Math.max(v, 1e-6) })),
      borderColor: colores[mod].tint,
      backgroundColor: colores[mod].tint,
      pointRadius: 0,
      borderDash: [4, 4],
      fill: false,
    });
  }

  estado.graficoMC = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: 'linear', title: { display: true, text: 'SNR (dB)' } },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'BER (log)' },
          min: 1e-5,
          max: 1,
        },
      },
      plugins: {
        legend: {
          labels: { filter: (item) => !item.text.includes('IC95') },
        },
        title: {
          display: true,
          text: 'BER vs SNR — Monte Carlo (10 corridas, IC 95% T-Student)',
        },
      },
    },
  });
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

  actualizarUE();
  actualizarParametros();
  $('btn-simular').disabled = true;
});
