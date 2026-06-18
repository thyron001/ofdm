// === Simulador OFDM / SC-FDM — LTE — Frontend ===
// Una sola factoría crearPanel(cfg) construye cada pestaña (OFDM y SC-FDM) a
// partir de un mapa de IDs. Toda la lógica vive en el closure: no hay estado global.

// ----- Constantes compartidas -----
const CANVAS_TAM = 500;
const CENTRO = { x: CANVAS_TAM / 2, y: CANVAS_TAM / 2 };
const RADIO_MAX_M = 1500;
const RADIO_MAX_PX = 230;
const M_POR_PX = RADIO_MAX_M / RADIO_MAX_PX;

const COMBINACIONES_NO_VALIDAS = new Set(['1.4_7.5', '3_7.5']);

const COLORES_MOD = { 'QPSK': '#18A34B', '16-QAM': '#1E54E0', '64-QAM': '#D82A2A' };

// Zonas de cobertura por esquema
const ZONAS_OFDM = [
  { rMin: 0,   rMax: 300,  mod: '64-QAM', color: 'rgba(24,163,75,0.18)',  borde: '#18A34B' },
  { rMin: 300, rMax: 800,  mod: '16-QAM', color: 'rgba(30,84,224,0.14)',  borde: '#1E54E0' },
  { rMin: 800, rMax: 1500, mod: 'QPSK',   color: 'rgba(229,138,26,0.16)', borde: '#E58A1A' },
];
const ZONAS_SCFDM = [
  { rMin: 0,   rMax: 600,  mod: '16-QAM', color: 'rgba(30,84,224,0.14)',  borde: '#1E54E0' },
  { rMin: 600, rMax: 1500, mod: 'QPSK',   color: 'rgba(229,138,26,0.16)', borde: '#E58A1A' },
];

function badgeClasePorMod(mod) {
  return mod === '64-QAM' ? 'badge-verde' : mod === '16-QAM' ? 'badge-azul' : 'badge-naranja';
}
function formatearTiempo(s) {
  if (s < 1e-3) return (s * 1e6).toFixed(2) + ' µs';
  if (s < 1) return (s * 1e3).toFixed(3) + ' ms';
  return s.toFixed(3) + ' s';
}

// Plugin Chart.js: barras de error (IC 95%) en cada punto.
// Para distinguir OFDM y SC-FDM de la MISMA modulación (mismo color), cuando el gráfico
// mezcla ambos esquemas se usan estilos distintos y un pequeño desplazamiento horizontal:
//   - OFDM   : barra sólida, gruesa, topes anchos, desplazada a la izquierda.
//   - SC-FDM : barra discontinua, fina, topes angostos, desplazada a la derecha.
const PLUGIN_ERROR_BARS = {
  id: 'errorBars',
  afterDatasetsDraw(chart) {
    const { ctx, scales: { x, y } } = chart;
    const mixto = chart.data.datasets.some((d) => d.esquema === 'SC-FDM') &&
                  chart.data.datasets.some((d) => d.esquema === 'OFDM');
    chart.data.datasets.forEach((ds, idx) => {
      if (!ds.errorBars) return;
      const meta = chart.getDatasetMeta(idx);
      if (meta.hidden) return;
      const esSC = ds.esquema === 'SC-FDM';
      ctx.save();
      ctx.strokeStyle = ds.borderColor;
      ctx.lineWidth = esSC ? 1.3 : 2.3;
      ctx.setLineDash(esSC ? [4, 3] : []);
      const off = mixto ? (esSC ? 4 : -4) : 0;   // dodge solo si hay ambos esquemas
      const w = esSC ? 4 : 8;                      // ancho de los topes
      ds.data.forEach((pt, i) => {
        const eb = ds.errorBars[i];
        if (!eb) return;
        const px = x.getPixelForValue(pt.x) + off;
        const pyMin = y.getPixelForValue(Math.max(eb.lo, y.min));
        const pyMax = y.getPixelForValue(Math.max(eb.hi, y.min));
        ctx.beginPath();
        ctx.moveTo(px, pyMin); ctx.lineTo(px, pyMax);
        ctx.moveTo(px - w, pyMin); ctx.lineTo(px + w, pyMin);
        ctx.moveTo(px - w, pyMax); ctx.lineTo(px + w, pyMax);
        ctx.stroke();
      });
      ctx.restore();   // restaura lineDash/lineWidth
    });
  },
};

// Lista de paneles para difundir la imagen subida a ambas pestañas
const paneles = [];
function broadcastImagen(info) { paneles.forEach((p) => p.notificarImagen(info)); }

// =====================================================================
// === FACTORÍA DE PANEL                                              ===
// =====================================================================
function crearPanel(cfg) {
  const $ = (k) => (cfg.ids[k] ? document.getElementById(cfg.ids[k]) : null);

  const estado = {
    imagenSubida: false,
    nBits: 0,
    modulacion: null,
    distancia: 0,
    uePos: { x: CENTRO.x, y: CENTRO.y },
    arrastrando: false,
    graficoMC: null,
    graficoPAPR: null,
  };

  // ---------- Parámetros ----------
  function validarCombinacion() {
    const bw = $('bw').value;
    const df = $('df').value;
    const aviso = $('avisoCombinacion');
    const ok = !COMBINACIONES_NO_VALIDAS.has(`${bw}_${df}`);
    if (aviso) aviso.style.display = ok ? 'none' : 'block';
    return ok;
  }

  function actualizarOpcionesCP() {
    const df = $('df').value;
    const cp = $('cp');
    if (df === '15') {
      cp.innerHTML = '<option value="normal">Normal (4.7 µs)</option>' +
                     '<option value="extendido">Extendido (16.67 µs)</option>';
    } else {
      cp.innerHTML = '<option value="extendido">Extendido (33.33 µs)</option>';
    }
  }

  async function actualizarParametros() {
    if (!validarCombinacion()) {
      $('kvParams').innerHTML = '<div class="kv"><div class="kv-label">Combinación inválida</div></div>';
      return;
    }
    const payload = {
      bw_mhz: parseFloat($('bw').value),
      delta_f_khz: parseFloat($('df').value),
      tipo_cp: $('cp').value,
      modulacion: estado.modulacion || cfg.modDefault,
      n_bits: estado.nBits,
    };
    try {
      const r = await fetch('/calcular_parametros', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (j.error) {
        $('kvParams').innerHTML = `<div class="kv"><div class="kv-label">Error</div><div class="kv-valor" style="font-size:13px">${j.error}</div></div>`;
        return;
      }
      renderKV(j);
    } catch (e) { console.error(e); }
  }

  function renderKV(p, extras = {}) {
    const esSC = cfg.esquema === 'SC-FDM';
    const items = [
      ['Subportadoras útiles (N_SC)', p.n_sc],
      ['Tamaño de la FFT (N_FFT)', p.n_fft],
      ['Frecuencia de muestreo (fs)', (p.fs / 1e6).toFixed(3) + ' MHz'],
      ['Muestras de prefijo cíclico (N_CP)', p.n_cp],
    ];
    if (esSC) {
      items.push(['Tamaño DFT-spread (M)', p.n_datos]);
      items.push(['Subportadoras de datos', p.n_datos]);
      items.push(['Mapeo de subportadoras', 'Localizado (contiguo)']);
    } else {
      items.push(['Subportadoras piloto', p.n_pilotos]);
      items.push(['Subportadoras de datos', p.n_datos]);
    }
    items.push(
      ['Bits totales de la imagen', (estado.nBits || 0).toLocaleString()],
      ['Símbolos QAM totales', p.n_simbolos_qam.toLocaleString()],
      [esSC ? 'Símbolos SC-FDM transmitidos' : 'Símbolos OFDM transmitidos', p.n_simbolos_ofdm.toLocaleString()],
      ['Duración de cada símbolo', p.duracion_simbolo_us.toFixed(2) + ' µs'],
      ['Duración del prefijo cíclico', p.duracion_cp_us.toFixed(2) + ' µs'],
      ['Modulación aplicada', estado.modulacion || '—'],
    );
    if (extras.tiempo_aire_s != null) {
      items.push(['Tiempo de transmisión (aire)', formatearTiempo(extras.tiempo_aire_s)]);
    }
    $('kvParams').innerHTML = items.map(([l, v]) =>
      `<div class="kv"><div class="kv-label">${l}</div><div class="kv-valor">${v}</div></div>`
    ).join('');
  }

  // ---------- SNR slider ----------
  function sincSNR() {
    $('snr').addEventListener('input', () => { $('snrNum').value = $('snr').value; });
    $('snrNum').addEventListener('input', () => {
      let v = parseInt($('snrNum').value || '0');
      if (v < 0) v = 0; if (v > 100) v = 100;
      $('snr').value = v;
    });
  }

  // ---------- Imagen ----------
  async function subirImagen(e) {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('imagen', file);
    $('infoImagen').textContent = 'Subiendo...';
    const r = await fetch('/subir_imagen', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.error) {
      $('infoImagen').textContent = 'Error: ' + j.error;
      $('infoImagen').className = 'estado err';
      return;
    }
    broadcastImagen(j); // actualiza esta pestaña y la otra
  }

  function notificarImagen(info) {
    estado.imagenSubida = true;
    estado.nBits = info.n_bits;
    if (info.preview_b64) {
      $('preview').src = info.preview_b64;
      $('preview').style.display = 'block';
    }
    $('infoImagen').className = 'estado ok';
    $('infoImagen').textContent =
      `${info.ancho}×${info.alto}, ${info.canales} canal(es), ${info.n_bits.toLocaleString()} bits ✓`;
    if (estado.modulacion) $('btnSimular').disabled = false;
    actualizarParametros();
  }

  // ---------- Canvas de cobertura ----------
  function dibujarCobertura() {
    const cv = $('canvasCobertura');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);

    ctx.strokeStyle = '#E4EAF0'; ctx.lineWidth = 1;
    for (let i = 50; i < CANVAS_TAM; i += 50) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, CANVAS_TAM); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(CANVAS_TAM, i); ctx.stroke();
    }
    [...cfg.zonas].reverse().forEach((z) => {
      const rPxMax = z.rMax / M_POR_PX;
      ctx.beginPath(); ctx.arc(CENTRO.x, CENTRO.y, rPxMax, 0, Math.PI * 2);
      ctx.fillStyle = z.color; ctx.fill();
      ctx.strokeStyle = z.borde; ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    });
    ctx.fillStyle = '#0B2540'; ctx.font = 'bold 12px Nunito'; ctx.textAlign = 'center';
    cfg.zonas.forEach((z) => {
      const rMidPx = ((z.rMin + z.rMax) / 2) / M_POR_PX;
      ctx.fillText(`${z.mod} (${z.rMin}–${z.rMax} m)`, CENTRO.x, CENTRO.y - rMidPx + 4);
    });
    // Antena
    ctx.fillStyle = '#1C3A58';
    ctx.beginPath();
    ctx.moveTo(CENTRO.x, CENTRO.y - 14);
    ctx.lineTo(CENTRO.x - 10, CENTRO.y + 8);
    ctx.lineTo(CENTRO.x + 10, CENTRO.y + 8);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#1E54E0';
    ctx.beginPath(); ctx.arc(CENTRO.x, CENTRO.y, 4, 0, Math.PI * 2); ctx.fill();
    // UE
    const fuera = estado.distancia > RADIO_MAX_M;
    ctx.fillStyle = fuera ? '#D82A2A' : '#1642B4';
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(estado.uePos.x, estado.uePos.y, 10, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
    // Escala
    ctx.fillStyle = '#5B6B7E'; ctx.font = 'bold 10px Nunito'; ctx.textAlign = 'left';
    ctx.fillText(`Escala: ${Math.round(M_POR_PX)} m/px`, 10, CANVAS_TAM - 12);
  }

  function calcDistancia() {
    const dx = estado.uePos.x - CENTRO.x;
    const dy = estado.uePos.y - CENTRO.y;
    return Math.round(Math.sqrt(dx * dx + dy * dy) * M_POR_PX);
  }
  function modulacionPorDistancia(d) {
    if (d > RADIO_MAX_M) return null;
    const z = cfg.zonas.find((z) => d >= z.rMin && d < z.rMax);
    return z ? z.mod : cfg.zonas[cfg.zonas.length - 1].mod;
  }
  function actualizarUE() {
    estado.distancia = calcDistancia();
    const mod = modulacionPorDistancia(estado.distancia);
    $('distanciaUe').textContent = estado.distancia;
    const badge = $('badgeModulacion');
    if (mod === null) {
      badge.textContent = 'Fuera de cobertura';
      badge.className = 'badge badge-rojo';
      estado.modulacion = null;
      $('btnSimular').disabled = true;
    } else {
      estado.modulacion = mod;
      badge.textContent = mod;
      badge.className = 'badge ' + badgeClasePorMod(mod);
      $('btnSimular').disabled = !estado.imagenSubida;
    }
    dibujarCobertura();
    actualizarParametros();
  }
  function bindCobertura() {
    const cv = $('canvasCobertura');
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
      estado.uePos = posMouse(ev); actualizarUE();
    });
    window.addEventListener('mouseup', () => { estado.arrastrando = false; });
  }

  // ---------- Simulación ----------
  async function ejecutarSimulacion() {
    if (!estado.imagenSubida) { alert('Sube una imagen primero'); return; }
    if (!estado.modulacion) { alert('UE fuera de cobertura'); return; }

    $('btnSimular').disabled = true;
    $('btnSimular').innerHTML = '<span class="spinner"></span> Simulando...';
    $('estadoSim').textContent = `Ejecutando cadena ${cfg.esquema}: TX → Canal → AWGN → RX...`;
    $('estadoSim').className = 'estado';

    const payload = {
      bw_mhz: parseFloat($('bw').value),
      delta_f_khz: parseFloat($('df').value),
      tipo_cp: $('cp').value,
      snr_db: parseFloat($('snr').value),
      modulacion: estado.modulacion,
      esquema: cfg.esquema,
    };
    try {
      const r = await fetch('/simular', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      pintarResultados(j);
      $('estadoSim').textContent = 'Simulación completada ✓';
      $('estadoSim').className = 'estado ok';
    } catch (e) {
      $('estadoSim').textContent = 'Error: ' + e.message;
      $('estadoSim').className = 'estado err';
    } finally {
      $('btnSimular').disabled = false;
      $('btnSimular').textContent = 'Simular Transmisión';
    }
  }

  function pintarResultados(j) {
    $('zonaResultados').style.display = 'block';
    $('imgTx').src = j.imagen_original_b64;
    $('imgRx').src = j.imagen_recuperada_b64;
    $('berValor').textContent = (j.ber * 100).toFixed(3) + ' %';
    $('berDetalle').textContent =
      `${j.bits_erroneos.toLocaleString()} / ${j.bits_transmitidos.toLocaleString()} bits — ${j.n_simbolos_ofdm} símbolos ${j.esquema} — SNR ${j.snr_db} dB`;
    dibujarConstelacion(j.constelacion_tx, j.constelacion_rx, j.modulacion);
    dibujarMapaSC(j.mapa_subportadoras);
    renderKV(j.parametros, { tiempo_aire_s: j.tiempo_aire_s });
    $('tituloConstelacion').textContent = `Constelación RX — ${j.esquema} ${j.modulacion}`;
    const mapa = j.mapa_subportadoras;
    $('resumenSc').textContent = (cfg.esquema === 'SC-FDM')
      ? `Bloque de datos contiguo: ${mapa.datos} · Guarda: ${mapa.guarda} · Total: ${mapa.total} (mapeo localizado, sin pilotos intercalados)`
      : `Datos: ${mapa.datos} · Pilotos: ${mapa.pilotos} · Total: ${mapa.total}`;
    if (cfg.mostrarFormaOnda && j.forma_onda) dibujarFormaOnda(j.forma_onda);
  }

  function dibujarConstelacion(tx, rx, mod) {
    const cv = $('constelacionCanvas');
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const lim = mod === 'QPSK' ? 1.2 : (mod === '16-QAM' ? 1.4 : 1.6);
    const sx = (x) => W / 2 + (x / lim) * (W / 2 - 10);
    const sy = (y) => H / 2 - (y / lim) * (H / 2 - 10);
    ctx.strokeStyle = '#E4EAF0'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H);
    ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
    ctx.fillStyle = 'rgba(24,163,75,0.45)';
    for (let i = 0; i < rx.real.length; i++) {
      ctx.beginPath(); ctx.arc(sx(rx.real[i]), sy(rx.imag[i]), 2, 0, Math.PI * 2); ctx.fill();
    }
    const dedupe = new Set(); const claves = [];
    for (let i = 0; i < tx.real.length; i++) {
      const k = `${tx.real[i].toFixed(2)}|${tx.imag[i].toFixed(2)}`;
      if (!dedupe.has(k)) { dedupe.add(k); claves.push([tx.real[i], tx.imag[i]]); }
    }
    ctx.strokeStyle = '#D82A2A'; ctx.lineWidth = 2;
    claves.forEach(([r, i]) => {
      const x = sx(r), y = sy(i);
      ctx.beginPath();
      ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5);
      ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5);
      ctx.stroke();
    });
  }

  function dibujarMapaSC(mapa) {
    const cv = $('mapaScCanvas');
    const ctx = cv.getContext('2d');
    const W = cv.clientWidth || 800;
    cv.width = W; cv.height = 120;
    ctx.clearRect(0, 0, W, 120);
    const total = mapa.total;
    const ancho = W / total;
    const pilotosSet = new Set(mapa.indices_pilotos || []);
    const guardaSet = new Set(mapa.indices_guarda || []);
    for (let k = 0; k < total; k++) {
      let color = '#1E54E0';           // datos
      if (pilotosSet.has(k)) color = '#D82A2A';      // pilotos
      else if (guardaSet.has(k)) color = '#C9D4E0';  // guarda
      ctx.fillStyle = color;
      ctx.fillRect(k * ancho, 10, Math.max(ancho - 0.5, 1), 90);
    }
    // Leyenda
    ctx.font = 'bold 11px Nunito';
    ctx.fillStyle = '#1E54E0'; ctx.fillRect(4, 107, 10, 8);
    ctx.fillStyle = '#5B6B7E'; ctx.fillText('Datos', 18, 115);
    if (cfg.esquema === 'SC-FDM') {
      ctx.fillStyle = '#C9D4E0'; ctx.fillRect(80, 107, 10, 8);
      ctx.fillStyle = '#5B6B7E'; ctx.fillText('Guarda', 94, 115);
    } else {
      ctx.fillStyle = '#D82A2A'; ctx.fillRect(80, 107, 10, 8);
      ctx.fillStyle = '#5B6B7E'; ctx.fillText('Pilotos (cada 12)', 94, 115);
    }
  }

  // ---------- Forma de onda |x(t)| (solo SC-FDM) ----------
  function dibujarFormaOnda(fo) {
    const cv = $('formaOndaCanvas');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    const W = cv.clientWidth || 1000; const H = 280;
    cv.width = W; cv.height = H;
    ctx.clearRect(0, 0, W, H);
    const padL = 46, padR = 12, padT = 14, padB = 28;
    const w = W - padL - padR, h = H - padT - padB;

    const eo = fo.ofdm.envolvente, es = fo.scfdm.envolvente;
    const n = Math.max(eo.length, es.length);
    const ymax = Math.max(Math.max(...eo), Math.max(...es)) * 1.08 || 1;
    const sx = (i) => padL + (i / (n - 1)) * w;
    const sy = (v) => padT + h - (v / ymax) * h;

    // Ejes
    ctx.strokeStyle = '#E4EAF0'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + h); ctx.lineTo(padL + w, padT + h); ctx.stroke();
    ctx.fillStyle = '#5B6B7E'; ctx.font = 'bold 10px Nunito'; ctx.textAlign = 'right';
    ctx.fillText(ymax.toFixed(2), padL - 6, padT + 8);
    ctx.fillText('0', padL - 6, padT + h);
    ctx.textAlign = 'center';
    ctx.fillText('muestras en el tiempo (un símbolo)', padL + w / 2, H - 6);
    ctx.save(); ctx.translate(12, padT + h / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('|x(t)|', 0, 0); ctx.restore();

    const traza = (env, color) => {
      ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.beginPath();
      for (let i = 0; i < env.length; i++) {
        const px = sx(i), py = sy(env[i]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    };
    traza(eo, 'rgba(216,42,42,0.85)');   // OFDM rojo
    traza(es, 'rgba(30,84,224,0.95)');   // SC-FDM azul

    // Leyenda
    ctx.textAlign = 'left'; ctx.font = 'bold 12px Nunito';
    ctx.fillStyle = '#D82A2A'; ctx.fillRect(padL + 8, padT + 4, 14, 4);
    ctx.fillStyle = '#5B6B7E'; ctx.fillText(`OFDM  (PAPR ${fo.ofdm.papr_db.toFixed(2)} dB)`, padL + 28, padT + 10);
    ctx.fillStyle = '#1E54E0'; ctx.fillRect(padL + 8, padT + 22, 14, 4);
    ctx.fillStyle = '#5B6B7E'; ctx.fillText(`SC-FDM  (PAPR ${fo.scfdm.papr_db.toFixed(2)} dB)`, padL + 28, padT + 28);

    const dif = (fo.ofdm.papr_db - fo.scfdm.papr_db);
    $('resumenPapr') && ($('resumenPapr').textContent =
      `SC-FDM reduce el PAPR de este símbolo en ${dif.toFixed(2)} dB respecto a OFDM (forma de onda más plana = menor relación pico/promedio).`);
  }

  // ---------- Monte Carlo (genérico: 3 o 5 series) ----------
  async function ejecutarMontecarlo() {
    $('btnMc').disabled = true;
    $('btnMc').innerHTML = '<span class="spinner"></span> Ejecutando Monte Carlo...';
    const nCurvas = cfg.esquemasMC.reduce((a, e) => a + (e === 'OFDM' ? 3 : 2), 0);
    $('estadoSim').textContent = `Monte Carlo: ${nCurvas} curvas × 16 SNR × 10 corridas (puede tardar)...`;
    $('estadoSim').className = 'estado';

    const payload = {
      bw_mhz: parseFloat($('bw').value),
      delta_f_khz: parseFloat($('df').value),
      tipo_cp: $('cp').value,
      esquemas: cfg.esquemasMC,
    };
    try {
      const r = await fetch('/montecarlo', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      pintarMC(j);
      $('estadoSim').textContent = 'Monte Carlo completado ✓';
      $('estadoSim').className = 'estado ok';
    } catch (e) {
      $('estadoSim').textContent = 'Error: ' + e.message;
      $('estadoSim').className = 'estado err';
    } finally {
      $('btnMc').disabled = false;
      $('btnMc').textContent = cfg.esquema === 'SC-FDM' ? 'Monte Carlo (OFDM vs SC-FDM)' : 'Simulación Monte Carlo';
    }
  }

  function pintarMC(j) {
    $('modal').style.display = 'flex';
    if (estado.graficoMC) estado.graficoMC.destroy();
    if (estado.graficoPAPR) estado.graficoPAPR.destroy();
    const PISO = 1e-5;

    // BER vs SNR
    const datasets = j.series_ber.map((s) => ({
      label: `${s.esquema} ${s.modulacion}`,
      esquema: s.esquema,
      data: s.ber_promedio.map((v, i) => ({ x: j.snr_valores[i], y: Math.max(v, PISO) })),
      errorBars: s.ber_promedio.map((_, i) => ({
        lo: Math.max(s.ic_inferior[i], PISO), hi: Math.max(s.ic_superior[i], PISO),
      })),
      borderColor: COLORES_MOD[s.modulacion],
      backgroundColor: COLORES_MOD[s.modulacion],
      borderDash: s.esquema === 'SC-FDM' ? [6, 4] : [],
      pointStyle: s.esquema === 'SC-FDM' ? 'triangle' : 'circle',
      pointRadius: 5, tension: 0.15, fill: false,
    }));

    estado.graficoMC = new Chart($('montecarloCanvas').getContext('2d'), {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { type: 'linear', title: { display: true, text: 'SNR (dB)', font: { weight: 700 } },
               ticks: { stepSize: 1 }, min: -0.5, max: 15.5 },
          y: { type: 'logarithmic', title: { display: true, text: 'BER (escala logarítmica)', font: { weight: 700 } },
               min: PISO, max: 1 },
        },
        plugins: {
          legend: { position: 'top', labels: { font: { weight: 700 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const eb = ctx.dataset.errorBars[ctx.dataIndex];
                return `${ctx.dataset.label}: BER=${ctx.parsed.y.toExponential(2)}  IC95% [${eb.lo.toExponential(2)}, ${eb.hi.toExponential(2)}]`;
              },
            },
          },
        },
      },
      plugins: [PLUGIN_ERROR_BARS],
    });

    // CCDF del PAPR
    if (j.papr) {
      const x = j.papr.x_db;
      const dsP = j.papr.series.map((s) => ({
        label: `${s.esquema} ${s.modulacion}`,
        data: s.ccdf.map((v, i) => ({ x: x[i], y: Math.max(v, 1e-4) })),
        borderColor: COLORES_MOD[s.modulacion],
        backgroundColor: COLORES_MOD[s.modulacion],
        borderDash: s.esquema === 'SC-FDM' ? [6, 4] : [],
        pointRadius: 0, borderWidth: 2, tension: 0.2, fill: false,
      }));
      estado.graficoPAPR = new Chart($('paprCanvas').getContext('2d'), {
        type: 'line',
        data: { datasets: dsP },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          scales: {
            x: { type: 'linear', title: { display: true, text: 'PAPR_0 (dB)', font: { weight: 700 } },
                 min: 0, max: Math.max(...x) },
            y: { type: 'logarithmic', title: { display: true, text: 'Pr{PAPR > PAPR_0}', font: { weight: 700 } },
                 min: 1e-3, max: 1 },
          },
          plugins: {
            legend: { position: 'top', labels: { font: { weight: 700 } } },
            title: { display: true, text: `CCDF del PAPR (${j.papr.n_simbolos} símbolos por curva)` },
          },
        },
      });
    }
  }

  function descargarCanvas(canvasKey, prefijo) {
    const src = $(canvasKey);
    const tmp = document.createElement('canvas');
    tmp.width = src.width; tmp.height = src.height;
    const tctx = tmp.getContext('2d');
    tctx.fillStyle = '#FFFFFF'; tctx.fillRect(0, 0, tmp.width, tmp.height);
    tctx.drawImage(src, 0, 0);
    const link = document.createElement('a');
    link.download = `${prefijo}_${Date.now()}.png`;
    link.href = tmp.toDataURL('image/png');
    link.click();
  }
  function descargarMC() {
    const pref = cfg.esquema === 'SC-FDM' ? 'comparacion' : 'ofdm';
    if (estado.graficoMC) descargarCanvas('montecarloCanvas', `${pref}_ber_snr`);
    if (estado.graficoPAPR) descargarCanvas('paprCanvas', `${pref}_papr_ccdf`);
  }

  // ---------- Cableado de eventos ----------
  function bind() {
    sincSNR();
    bindCobertura();
    $('df').addEventListener('change', () => { actualizarOpcionesCP(); actualizarParametros(); validarCombinacion(); });
    $('bw').addEventListener('change', () => { actualizarParametros(); validarCombinacion(); });
    $('cp').addEventListener('change', actualizarParametros);
    $('archivo').addEventListener('change', subirImagen);
    $('btnSimular').addEventListener('click', ejecutarSimulacion);
    $('btnMc').addEventListener('click', ejecutarMontecarlo);

    // Lightbox al hacer clic en imágenes
    ['imgTx', 'imgRx'].forEach((k) => {
      $(k).addEventListener('click', () => {
        const lb = document.getElementById('lightbox');
        document.getElementById('lightbox-img').src = $(k).src;
        lb.style.display = 'flex';
      });
    });

    // Modal Monte Carlo
    $('btnCerrarMc').addEventListener('click', () => { $('modal').style.display = 'none'; });
    $('btnDescargarMc').addEventListener('click', descargarMC);
    $('modal').addEventListener('click', (ev) => {
      if (ev.target.id === cfg.ids.modal) $('modal').style.display = 'none';
    });

    actualizarUE();
    actualizarParametros();
    $('btnSimular').disabled = true;
  }

  return { bind, notificarImagen, estado, cerrarModal: () => { $('modal').style.display = 'none'; } };
}

// =====================================================================
// === PANEL DIVERSIDAD RX (MRC) — Práctica 3                        ===
// =====================================================================
// Panel propio (la factoría crearPanel está acoplada al mapa de cobertura, que aquí no aplica).
// El TX es OFDM; el RX combina N antenas con MRC. Color de curva = nº de antenas.
const COLORES_ANTENAS = { 2: '#18A34B', 4: '#1E54E0', 8: '#D82A2A' };

function crearPanelMRC() {
  const $ = (id) => document.getElementById(id);
  const estado = { imagenSubida: false, nBits: 0, graficoMC: null };

  // ---------- Parámetros ----------
  function validarCombinacion() {
    const ok = !COMBINACIONES_NO_VALIDAS.has(`${$('mrc-bw').value}_${$('mrc-df').value}`);
    $('mrc-aviso-combinacion').style.display = ok ? 'none' : 'block';
    return ok;
  }
  function actualizarOpcionesCP() {
    const df = $('mrc-df').value;
    const cp = $('mrc-cp');
    if (df === '15') {
      cp.innerHTML = '<option value="normal">Normal (4.7 µs)</option>' +
                     '<option value="extendido">Extendido (16.67 µs)</option>';
    } else {
      cp.innerHTML = '<option value="extendido">Extendido (33.33 µs)</option>';
    }
  }
  async function actualizarParametros() {
    if (!validarCombinacion()) {
      $('mrc-kv-params').innerHTML = '<div class="kv"><div class="kv-label">Combinación inválida</div></div>';
      return;
    }
    const payload = {
      bw_mhz: parseFloat($('mrc-bw').value),
      delta_f_khz: parseFloat($('mrc-df').value),
      tipo_cp: $('mrc-cp').value,
      modulacion: $('mrc-modulacion').value,
      n_bits: estado.nBits,
    };
    try {
      const r = await fetch('/calcular_parametros', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (j.error) {
        $('mrc-kv-params').innerHTML = `<div class="kv"><div class="kv-label">Error</div><div class="kv-valor" style="font-size:13px">${j.error}</div></div>`;
        return;
      }
      renderKV(j);
    } catch (e) { console.error(e); }
  }
  function renderKV(p) {
    const items = [
      ['Subportadoras útiles (N_SC)', p.n_sc],
      ['Tamaño de la FFT (N_FFT)', p.n_fft],
      ['Frecuencia de muestreo (fs)', (p.fs / 1e6).toFixed(3) + ' MHz'],
      ['Muestras de prefijo cíclico (N_CP)', p.n_cp],
      ['Subportadoras piloto', p.n_pilotos],
      ['Subportadoras de datos', p.n_datos],
      ['Bits totales de la imagen', (estado.nBits || 0).toLocaleString()],
      ['Símbolos QAM totales', p.n_simbolos_qam.toLocaleString()],
      ['Símbolos OFDM transmitidos', p.n_simbolos_ofdm.toLocaleString()],
      ['Modulación aplicada', $('mrc-modulacion').value],
      ['Antenas RX (demo)', $('mrc-antenas').value],
    ];
    $('mrc-kv-params').innerHTML = items.map(([l, v]) =>
      `<div class="kv"><div class="kv-label">${l}</div><div class="kv-valor">${v}</div></div>`
    ).join('');
  }

  // ---------- SNR slider ----------
  function sincSNR() {
    $('mrc-snr').addEventListener('input', () => { $('mrc-snr-num').value = $('mrc-snr').value; });
    $('mrc-snr-num').addEventListener('input', () => {
      let v = parseInt($('mrc-snr-num').value || '0');
      if (v < 0) v = 0; if (v > 100) v = 100;
      $('mrc-snr').value = v;
    });
  }

  // ---------- Imagen ----------
  async function subirImagen(e) {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('imagen', file);
    $('mrc-info-imagen').textContent = 'Subiendo...';
    const r = await fetch('/subir_imagen', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.error) {
      $('mrc-info-imagen').textContent = 'Error: ' + j.error;
      $('mrc-info-imagen').className = 'estado err';
      return;
    }
    broadcastImagen(j);   // actualiza esta pestaña y las otras
  }
  function notificarImagen(info) {
    estado.imagenSubida = true;
    estado.nBits = info.n_bits;
    if (info.preview_b64) {
      $('mrc-preview-imagen').src = info.preview_b64;
      $('mrc-preview-imagen').style.display = 'block';
    }
    $('mrc-info-imagen').className = 'estado ok';
    $('mrc-info-imagen').textContent =
      `${info.ancho}×${info.alto}, ${info.canales} canal(es), ${info.n_bits.toLocaleString()} bits ✓`;
    $('mrc-btn-simular').disabled = false;
    actualizarParametros();
  }

  // ---------- Nube de constelación RX ----------
  function dibujarNube(canvasId, puntos, mod, color) {
    const cv = $(canvasId);
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const lim = mod === 'QPSK' ? 1.6 : (mod === '16-QAM' ? 1.8 : 2.0);
    const sx = (x) => W / 2 + (x / lim) * (W / 2 - 10);
    const sy = (y) => H / 2 - (y / lim) * (H / 2 - 10);
    ctx.strokeStyle = '#E4EAF0'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H);
    ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
    ctx.fillStyle = color;
    for (let i = 0; i < puntos.real.length; i++) {
      ctx.beginPath(); ctx.arc(sx(puntos.real[i]), sy(puntos.imag[i]), 1.6, 0, Math.PI * 2); ctx.fill();
    }
  }

  // ---------- Simulación única (imagen + constelaciones antes/después) ----------
  async function ejecutarSimulacion() {
    if (!estado.imagenSubida) { alert('Sube una imagen primero'); return; }
    $('mrc-btn-simular').disabled = true;
    $('mrc-btn-simular').innerHTML = '<span class="spinner"></span> Simulando...';
    $('mrc-estado-sim').textContent = 'Ejecutando TX OFDM → N canales → MRC...';
    $('mrc-estado-sim').className = 'estado';
    const payload = {
      bw_mhz: parseFloat($('mrc-bw').value),
      delta_f_khz: parseFloat($('mrc-df').value),
      tipo_cp: $('mrc-cp').value,
      snr_db: parseFloat($('mrc-snr').value),
      modulacion: $('mrc-modulacion').value,
      n_rx: parseInt($('mrc-antenas').value),
    };
    try {
      const r = await fetch('/simular_mrc', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      pintarResultados(j);
      $('mrc-estado-sim').textContent = 'Simulación completada ✓';
      $('mrc-estado-sim').className = 'estado ok';
    } catch (e) {
      $('mrc-estado-sim').textContent = 'Error: ' + e.message;
      $('mrc-estado-sim').className = 'estado err';
    } finally {
      $('mrc-btn-simular').disabled = false;
      $('mrc-btn-simular').textContent = 'Simular Transmisión (MRC)';
    }
  }
  function pintarResultados(j) {
    $('mrc-zona-resultados').style.display = 'block';
    $('mrc-img-tx').src = j.imagen_original_b64;
    $('mrc-img-rx').src = j.imagen_recuperada_b64;
    $('mrc-ber-valor').textContent = (j.ber * 100).toFixed(3) + ' %';
    $('mrc-ber-detalle').textContent =
      `${j.bits_erroneos.toLocaleString()} / ${j.bits_transmitidos.toLocaleString()} bits — ${j.n_simbolos_ofdm} símbolos OFDM — ${j.n_rx} antenas — SNR ${j.snr_db} dB`;
    $('mrc-titulo-despues').textContent = `${j.n_rx} antenas (MRC)`;
    if (j.constelacion_rx_antes) {
      dibujarNube('mrc-constelacion-antes-canvas', j.constelacion_rx_antes, j.modulacion, 'rgba(216,42,42,0.45)');
    }
    dibujarNube('mrc-constelacion-despues-canvas', j.constelacion_rx, j.modulacion, 'rgba(24,163,75,0.5)');
    $('mrc-resumen-constelacion').textContent =
      `Izquierda: 1 sola antena con ZF (dispersa por el ruido). Derecha: las ${j.n_rx} antenas combinadas con MRC (nube más compacta = menor error).`;
    renderKV(j.parametros);
  }

  // ---------- Monte Carlo (una curva por nº de antenas) ----------
  async function ejecutarMontecarlo() {
    $('mrc-btn-mc').disabled = true;
    $('mrc-btn-mc').innerHTML = '<span class="spinner"></span> Ejecutando Monte Carlo...';
    $('mrc-estado-sim').textContent = 'Monte Carlo: 3 curvas (2/4/8 antenas) × 16 SNR × 8 corridas (puede tardar)...';
    $('mrc-estado-sim').className = 'estado';
    const payload = {
      bw_mhz: parseFloat($('mrc-bw').value),
      delta_f_khz: parseFloat($('mrc-df').value),
      tipo_cp: $('mrc-cp').value,
      modulacion: $('mrc-modulacion').value,
      antenas: [2, 4, 8],
    };
    try {
      const r = await fetch('/montecarlo_mrc', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      pintarMC(j);
      $('mrc-estado-sim').textContent = 'Monte Carlo completado ✓';
      $('mrc-estado-sim').className = 'estado ok';
    } catch (e) {
      $('mrc-estado-sim').textContent = 'Error: ' + e.message;
      $('mrc-estado-sim').className = 'estado err';
    } finally {
      $('mrc-btn-mc').disabled = false;
      $('mrc-btn-mc').textContent = 'Monte Carlo (MRC)';
    }
  }
  function pintarMC(j) {
    $('mrc-modal-mc').style.display = 'flex';
    $('mrc-mc-titulo').textContent = `Diversidad RX (MRC) — BER vs SNR — ${j.modulacion}`;
    if (estado.graficoMC) estado.graficoMC.destroy();
    const PISO = 1e-5;
    const datasets = j.series_ber.map((s) => ({
      label: `${s.n_rx} antenas`,
      data: s.ber_promedio.map((v, i) => ({ x: j.snr_valores[i], y: Math.max(v, PISO) })),
      errorBars: s.ber_promedio.map((_, i) => ({
        lo: Math.max(s.ic_inferior[i], PISO), hi: Math.max(s.ic_superior[i], PISO),
      })),
      borderColor: COLORES_ANTENAS[s.n_rx] || '#1E54E0',
      backgroundColor: COLORES_ANTENAS[s.n_rx] || '#1E54E0',
      pointRadius: 5, tension: 0.15, fill: false,
    }));
    estado.graficoMC = new Chart($('mrc-montecarlo-canvas').getContext('2d'), {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { type: 'linear', title: { display: true, text: 'SNR por antena (dB)', font: { weight: 700 } },
               ticks: { stepSize: 1 }, min: -0.5, max: 15.5 },
          y: { type: 'logarithmic', title: { display: true, text: 'BER (escala logarítmica)', font: { weight: 700 } },
               min: PISO, max: 1 },
        },
        plugins: {
          legend: { position: 'top', labels: { font: { weight: 700 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const eb = ctx.dataset.errorBars[ctx.dataIndex];
                return `${ctx.dataset.label}: BER=${ctx.parsed.y.toExponential(2)}  IC95% [${eb.lo.toExponential(2)}, ${eb.hi.toExponential(2)}]`;
              },
            },
          },
        },
      },
      plugins: [PLUGIN_ERROR_BARS],
    });
  }

  // ---------- Descargar PNG ----------
  function descargarMC() {
    if (!estado.graficoMC) return;
    const src = $('mrc-montecarlo-canvas');
    const tmp = document.createElement('canvas');
    tmp.width = src.width; tmp.height = src.height;
    const tctx = tmp.getContext('2d');
    tctx.fillStyle = '#FFFFFF'; tctx.fillRect(0, 0, tmp.width, tmp.height);
    tctx.drawImage(src, 0, 0);
    const link = document.createElement('a');
    link.download = `mrc_ber_snr_${Date.now()}.png`;
    link.href = tmp.toDataURL('image/png');
    link.click();
  }

  // ---------- Cableado ----------
  function bind() {
    sincSNR();
    $('mrc-df').addEventListener('change', () => { actualizarOpcionesCP(); actualizarParametros(); validarCombinacion(); });
    $('mrc-bw').addEventListener('change', () => { actualizarParametros(); validarCombinacion(); });
    $('mrc-cp').addEventListener('change', actualizarParametros);
    $('mrc-modulacion').addEventListener('change', actualizarParametros);
    $('mrc-antenas').addEventListener('change', actualizarParametros);
    $('mrc-archivo-imagen').addEventListener('change', subirImagen);
    $('mrc-btn-simular').addEventListener('click', ejecutarSimulacion);
    $('mrc-btn-mc').addEventListener('click', ejecutarMontecarlo);
    ['mrc-img-tx', 'mrc-img-rx'].forEach((k) => {
      $(k).addEventListener('click', () => {
        const lb = document.getElementById('lightbox');
        document.getElementById('lightbox-img').src = $(k).src;
        lb.style.display = 'flex';
      });
    });
    $('mrc-btn-cerrar-mc').addEventListener('click', () => { $('mrc-modal-mc').style.display = 'none'; });
    $('mrc-btn-descargar-mc').addEventListener('click', descargarMC);
    $('mrc-modal-mc').addEventListener('click', (ev) => {
      if (ev.target.id === 'mrc-modal-mc') $('mrc-modal-mc').style.display = 'none';
    });
    actualizarParametros();
    $('mrc-btn-simular').disabled = true;
  }

  return { bind, notificarImagen, cerrarModal: () => { $('mrc-modal-mc').style.display = 'none'; } };
}

// =====================================================================
// === PANEL DIVERSIDAD TX (SFBC) — Práctica 4                       ===
// =====================================================================
// Panel propio (espejo de crearPanelMRC). El TX usa 2 antenas con código de Alamouti
// espacio-frecuencia; el color de cada curva identifica la configuración (1×1/2×1/2×2).
const COLORES_CONFIG = { '1x1': '#D82A2A', '2x1': '#1E54E0', '2x2': '#18A34B' };
const ETIQUETAS_CONFIG = { '1x1': '1×1 (sin diversidad)', '2x1': '2×1 (SFBC)', '2x2': '2×2 (SFBC)' };

function crearPanelSFBC() {
  const $ = (id) => document.getElementById(id);
  const estado = { imagenSubida: false, nBits: 0, graficoMC: null };

  // ---------- Parámetros ----------
  function validarCombinacion() {
    const ok = !COMBINACIONES_NO_VALIDAS.has(`${$('sfbc-bw').value}_${$('sfbc-df').value}`);
    $('sfbc-aviso-combinacion').style.display = ok ? 'none' : 'block';
    return ok;
  }
  function actualizarOpcionesCP() {
    const df = $('sfbc-df').value;
    const cp = $('sfbc-cp');
    if (df === '15') {
      cp.innerHTML = '<option value="normal">Normal (4.7 µs)</option>' +
                     '<option value="extendido">Extendido (16.67 µs)</option>';
    } else {
      cp.innerHTML = '<option value="extendido">Extendido (33.33 µs)</option>';
    }
  }
  async function actualizarParametros() {
    if (!validarCombinacion()) {
      $('sfbc-kv-params').innerHTML = '<div class="kv"><div class="kv-label">Combinación inválida</div></div>';
      return;
    }
    const payload = {
      bw_mhz: parseFloat($('sfbc-bw').value),
      delta_f_khz: parseFloat($('sfbc-df').value),
      tipo_cp: $('sfbc-cp').value,
      modulacion: $('sfbc-modulacion').value,
      n_bits: estado.nBits,
    };
    try {
      const r = await fetch('/calcular_parametros', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (j.error) {
        $('sfbc-kv-params').innerHTML = `<div class="kv"><div class="kv-label">Error</div><div class="kv-valor" style="font-size:13px">${j.error}</div></div>`;
        return;
      }
      renderKV(j);
    } catch (e) { console.error(e); }
  }
  function renderKV(p) {
    const pares = Math.floor((p.n_sc || 0) / 2);
    const items = [
      ['Subportadoras útiles (N_SC)', p.n_sc],
      ['Tamaño de la FFT (N_FFT)', p.n_fft],
      ['Frecuencia de muestreo (fs)', (p.fs / 1e6).toFixed(3) + ' MHz'],
      ['Muestras de prefijo cíclico (N_CP)', p.n_cp],
      ['Pares Alamouti por símbolo (N_SC/2)', pares],
      ['Bits totales de la imagen', (estado.nBits || 0).toLocaleString()],
      ['Modulación aplicada', $('sfbc-modulacion').value],
      ['Configuración SFBC', $('sfbc-config').value.replace('x', '×')],
    ];
    $('sfbc-kv-params').innerHTML = items.map(([l, v]) =>
      `<div class="kv"><div class="kv-label">${l}</div><div class="kv-valor">${v}</div></div>`
    ).join('') +
      '<div class="kv" style="grid-column:1/-1"><div class="kv-label" style="font-size:12px">' +
      'SFBC usa todas las subportadoras en pares (sin pilotos intercalados), igual criterio que SC-FDM.' +
      '</div></div>';
  }

  // ---------- SNR slider ----------
  function sincSNR() {
    $('sfbc-snr').addEventListener('input', () => { $('sfbc-snr-num').value = $('sfbc-snr').value; });
    $('sfbc-snr-num').addEventListener('input', () => {
      let v = parseInt($('sfbc-snr-num').value || '0');
      if (v < 0) v = 0; if (v > 100) v = 100;
      $('sfbc-snr').value = v;
    });
  }

  // ---------- Imagen ----------
  async function subirImagen(e) {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('imagen', file);
    $('sfbc-info-imagen').textContent = 'Subiendo...';
    const r = await fetch('/subir_imagen', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.error) {
      $('sfbc-info-imagen').textContent = 'Error: ' + j.error;
      $('sfbc-info-imagen').className = 'estado err';
      return;
    }
    broadcastImagen(j);   // actualiza esta pestaña y las otras
  }
  function notificarImagen(info) {
    estado.imagenSubida = true;
    estado.nBits = info.n_bits;
    if (info.preview_b64) {
      $('sfbc-preview-imagen').src = info.preview_b64;
      $('sfbc-preview-imagen').style.display = 'block';
    }
    $('sfbc-info-imagen').className = 'estado ok';
    $('sfbc-info-imagen').textContent =
      `${info.ancho}×${info.alto}, ${info.canales} canal(es), ${info.n_bits.toLocaleString()} bits ✓`;
    $('sfbc-btn-simular').disabled = false;
    actualizarParametros();
  }

  // ---------- Nube de constelación RX ----------
  function dibujarNube(canvasId, puntos, mod, color) {
    const cv = $(canvasId);
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const lim = mod === 'QPSK' ? 1.6 : (mod === '16-QAM' ? 1.8 : 2.0);
    const sx = (x) => W / 2 + (x / lim) * (W / 2 - 10);
    const sy = (y) => H / 2 - (y / lim) * (H / 2 - 10);
    ctx.strokeStyle = '#E4EAF0'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H);
    ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
    ctx.fillStyle = color;
    for (let i = 0; i < puntos.real.length; i++) {
      ctx.beginPath(); ctx.arc(sx(puntos.real[i]), sy(puntos.imag[i]), 1.6, 0, Math.PI * 2); ctx.fill();
    }
  }

  // ---------- Simulación única (imagen + constelaciones sin diversidad vs SFBC) ----------
  async function ejecutarSimulacion() {
    if (!estado.imagenSubida) { alert('Sube una imagen primero'); return; }
    $('sfbc-btn-simular').disabled = true;
    $('sfbc-btn-simular').innerHTML = '<span class="spinner"></span> Simulando...';
    $('sfbc-estado-sim').textContent = 'Ejecutando TX SFBC (2 antenas) → canales → decodificador Alamouti...';
    $('sfbc-estado-sim').className = 'estado';
    const payload = {
      bw_mhz: parseFloat($('sfbc-bw').value),
      delta_f_khz: parseFloat($('sfbc-df').value),
      tipo_cp: $('sfbc-cp').value,
      snr_db: parseFloat($('sfbc-snr').value),
      modulacion: $('sfbc-modulacion').value,
      config: $('sfbc-config').value,
    };
    try {
      const r = await fetch('/simular_sfbc', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      pintarResultados(j);
      $('sfbc-estado-sim').textContent = 'Simulación completada ✓';
      $('sfbc-estado-sim').className = 'estado ok';
    } catch (e) {
      $('sfbc-estado-sim').textContent = 'Error: ' + e.message;
      $('sfbc-estado-sim').className = 'estado err';
    } finally {
      $('sfbc-btn-simular').disabled = false;
      $('sfbc-btn-simular').textContent = 'Simular Transmisión (SFBC)';
    }
  }
  function pintarResultados(j) {
    $('sfbc-zona-resultados').style.display = 'block';
    $('sfbc-img-tx').src = j.imagen_original_b64;
    $('sfbc-img-rx').src = j.imagen_recuperada_b64;
    $('sfbc-ber-valor').textContent = (j.ber * 100).toFixed(3) + ' %';
    const configTxt = (j.config || '').replace('x', '×');
    $('sfbc-ber-detalle').textContent =
      `${j.bits_erroneos.toLocaleString()} / ${j.bits_transmitidos.toLocaleString()} bits — ${j.n_simbolos_ofdm} símbolos OFDM — SFBC ${configTxt} — SNR ${j.snr_db} dB`;
    $('sfbc-titulo-despues').textContent = `SFBC ${configTxt}`;
    if (j.constelacion_rx_antes) {
      dibujarNube('sfbc-constelacion-antes-canvas', j.constelacion_rx_antes, j.modulacion, 'rgba(216,42,42,0.45)');
    }
    dibujarNube('sfbc-constelacion-despues-canvas', j.constelacion_rx, j.modulacion, 'rgba(24,163,75,0.5)');
    const beraTxt = (typeof j.ber_antes === 'number') ? ` (BER sin diversidad ≈ ${(j.ber_antes * 100).toFixed(3)} %)` : '';
    $('sfbc-resumen-constelacion').textContent =
      `Izquierda: 1 sola antena TX con ZF (dispersa por el desvanecimiento)${beraTxt}. Derecha: SFBC ${configTxt} (nube más compacta = menor error gracias a la diversidad en TX).`;
    renderKV(j.parametros);
  }

  // ---------- Monte Carlo (una curva por configuración) ----------
  async function ejecutarMontecarlo() {
    $('sfbc-btn-mc').disabled = true;
    $('sfbc-btn-mc').innerHTML = '<span class="spinner"></span> Ejecutando Monte Carlo...';
    $('sfbc-estado-sim').textContent = 'Monte Carlo: 3 curvas (1×1 / 2×1 / 2×2) × 16 SNR × 8 corridas (puede tardar)...';
    $('sfbc-estado-sim').className = 'estado';
    const payload = {
      bw_mhz: parseFloat($('sfbc-bw').value),
      delta_f_khz: parseFloat($('sfbc-df').value),
      tipo_cp: $('sfbc-cp').value,
      modulacion: $('sfbc-modulacion').value,
      configs: ['1x1', '2x1', '2x2'],
    };
    try {
      const r = await fetch('/montecarlo_sfbc', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      pintarMC(j);
      $('sfbc-estado-sim').textContent = 'Monte Carlo completado ✓';
      $('sfbc-estado-sim').className = 'estado ok';
    } catch (e) {
      $('sfbc-estado-sim').textContent = 'Error: ' + e.message;
      $('sfbc-estado-sim').className = 'estado err';
    } finally {
      $('sfbc-btn-mc').disabled = false;
      $('sfbc-btn-mc').textContent = 'Monte Carlo (SFBC)';
    }
  }
  function pintarMC(j) {
    $('sfbc-modal-mc').style.display = 'flex';
    $('sfbc-mc-titulo').textContent = `Diversidad TX (SFBC) — BER vs SNR — ${j.modulacion}`;
    if (estado.graficoMC) estado.graficoMC.destroy();
    const PISO = 1e-5;
    const datasets = j.series_ber.map((s) => ({
      label: ETIQUETAS_CONFIG[s.config] || s.config,
      data: s.ber_promedio.map((v, i) => ({ x: j.snr_valores[i], y: Math.max(v, PISO) })),
      errorBars: s.ber_promedio.map((_, i) => ({
        lo: Math.max(s.ic_inferior[i], PISO), hi: Math.max(s.ic_superior[i], PISO),
      })),
      borderColor: COLORES_CONFIG[s.config] || '#1E54E0',
      backgroundColor: COLORES_CONFIG[s.config] || '#1E54E0',
      pointRadius: 5, tension: 0.15, fill: false,
    }));
    estado.graficoMC = new Chart($('sfbc-montecarlo-canvas').getContext('2d'), {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { type: 'linear', title: { display: true, text: 'SNR recibida (dB)', font: { weight: 700 } },
               ticks: { stepSize: 1 }, min: -0.5, max: 15.5 },
          y: { type: 'logarithmic', title: { display: true, text: 'BER (escala logarítmica)', font: { weight: 700 } },
               min: PISO, max: 1 },
        },
        plugins: {
          legend: { position: 'top', labels: { font: { weight: 700 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const eb = ctx.dataset.errorBars[ctx.dataIndex];
                return `${ctx.dataset.label}: BER=${ctx.parsed.y.toExponential(2)}  IC95% [${eb.lo.toExponential(2)}, ${eb.hi.toExponential(2)}]`;
              },
            },
          },
        },
      },
      plugins: [PLUGIN_ERROR_BARS],
    });
  }

  // ---------- Descargar PNG ----------
  function descargarMC() {
    if (!estado.graficoMC) return;
    const src = $('sfbc-montecarlo-canvas');
    const tmp = document.createElement('canvas');
    tmp.width = src.width; tmp.height = src.height;
    const tctx = tmp.getContext('2d');
    tctx.fillStyle = '#FFFFFF'; tctx.fillRect(0, 0, tmp.width, tmp.height);
    tctx.drawImage(src, 0, 0);
    const link = document.createElement('a');
    link.download = `sfbc_ber_snr_${Date.now()}.png`;
    link.href = tmp.toDataURL('image/png');
    link.click();
  }

  // ---------- Cableado ----------
  function bind() {
    sincSNR();
    $('sfbc-df').addEventListener('change', () => { actualizarOpcionesCP(); actualizarParametros(); validarCombinacion(); });
    $('sfbc-bw').addEventListener('change', () => { actualizarParametros(); validarCombinacion(); });
    $('sfbc-cp').addEventListener('change', actualizarParametros);
    $('sfbc-modulacion').addEventListener('change', actualizarParametros);
    $('sfbc-config').addEventListener('change', actualizarParametros);
    $('sfbc-archivo-imagen').addEventListener('change', subirImagen);
    $('sfbc-btn-simular').addEventListener('click', ejecutarSimulacion);
    $('sfbc-btn-mc').addEventListener('click', ejecutarMontecarlo);
    ['sfbc-img-tx', 'sfbc-img-rx'].forEach((k) => {
      $(k).addEventListener('click', () => {
        const lb = document.getElementById('lightbox');
        document.getElementById('lightbox-img').src = $(k).src;
        lb.style.display = 'flex';
      });
    });
    $('sfbc-btn-cerrar-mc').addEventListener('click', () => { $('sfbc-modal-mc').style.display = 'none'; });
    $('sfbc-btn-descargar-mc').addEventListener('click', descargarMC);
    $('sfbc-modal-mc').addEventListener('click', (ev) => {
      if (ev.target.id === 'sfbc-modal-mc') $('sfbc-modal-mc').style.display = 'none';
    });
    actualizarParametros();
    $('sfbc-btn-simular').disabled = true;
  }

  return { bind, notificarImagen, cerrarModal: () => { $('sfbc-modal-mc').style.display = 'none'; } };
}

// =====================================================================
// === CONFIGURACIÓN DE LOS DOS PANELES                              ===
// =====================================================================
function idsConPrefijo(pre, extra = {}) {
  const base = {
    bw: 'bw', df: 'df', cp: 'cp', avisoCombinacion: 'aviso-combinacion',
    snr: 'snr', snrNum: 'snr-num',
    archivo: 'archivo-imagen', preview: 'preview-imagen', infoImagen: 'info-imagen',
    kvParams: 'kv-params',
    canvasCobertura: 'canvas-cobertura', distanciaUe: 'distancia-ue', badgeModulacion: 'badge-modulacion',
    btnSimular: 'btn-simular', btnMc: 'btn-mc', estadoSim: 'estado-sim',
    zonaResultados: 'zona-resultados', imgTx: 'img-tx', imgRx: 'img-rx',
    berValor: 'ber-valor', berDetalle: 'ber-detalle',
    constelacionCanvas: 'constelacion-canvas', tituloConstelacion: 'titulo-constelacion',
    mapaScCanvas: 'mapa-sc-canvas', resumenSc: 'resumen-sc',
    modal: 'modal-mc', montecarloCanvas: 'montecarlo-canvas', paprCanvas: 'papr-canvas',
    btnCerrarMc: 'btn-cerrar-mc', btnDescargarMc: 'btn-descargar-mc',
  };
  const out = {};
  for (const k in base) out[k] = `${pre}-${base[k]}`;
  return Object.assign(out, extra);
}

// =====================================================================
// === PESTAÑAS + INICIALIZACIÓN                                     ===
// =====================================================================
function initTabs() {
  const botones = document.querySelectorAll('.tab-btn');
  botones.forEach((b) => {
    b.addEventListener('click', () => {
      botones.forEach((x) => x.classList.remove('activa'));
      document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('activa'));
      b.classList.add('activa');
      document.getElementById('panel-' + b.dataset.tab).classList.add('activa');
    });
  });
}

function initLightbox() {
  const lb = document.getElementById('lightbox');
  lb.addEventListener('click', () => { lb.style.display = 'none'; });
}

window.addEventListener('DOMContentLoaded', () => {
  const panelOFDM = crearPanel({
    esquema: 'OFDM', modDefault: '16-QAM',
    ids: idsConPrefijo('ofdm'),
    zonas: ZONAS_OFDM, esquemasMC: ['OFDM'], mostrarFormaOnda: false,
  });
  const panelSCFDM = crearPanel({
    esquema: 'SC-FDM', modDefault: '16-QAM',
    ids: idsConPrefijo('scfdm', { formaOndaCanvas: 'scfdm-forma-onda-canvas', resumenPapr: 'scfdm-resumen-papr' }),
    zonas: ZONAS_SCFDM, esquemasMC: ['OFDM', 'SC-FDM'], mostrarFormaOnda: true,
  });
  const panelMRC = crearPanelMRC();
  const panelSFBC = crearPanelSFBC();
  paneles.push(panelOFDM, panelSCFDM, panelMRC, panelSFBC);
  panelOFDM.bind();
  panelSCFDM.bind();
  panelMRC.bind();
  panelSFBC.bind();
  initTabs();
  initLightbox();

  // Cerrar overlays con Escape
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') {
      document.getElementById('lightbox').style.display = 'none';
      panelOFDM.cerrarModal();
      panelSCFDM.cerrarModal();
      panelMRC.cerrarModal();
      panelSFBC.cerrarModal();
    }
  });
});
