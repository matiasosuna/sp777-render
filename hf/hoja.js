// Hoja de contactos de una lámina HyperFrames sin render: abre build/index.html en Chrome, salta a cada frase y saca un cuadro.
// Uso (desde la raíz del job): CARPETA=<nombre> node hoja.js  → cuadros/*.png, salida/errores.txt, salida/info.txt
const fs = require('fs'), path = require('path'), puppeteer = require(process.env.HOME + '/pp/node_modules/puppeteer');
(async () => {
  const dir = path.resolve('build'), html = fs.readFileSync(dir + '/index.html', 'utf8');
  const fams = new Set([...html.matchAll(/font-family:\s*["']([^"',;]+)["']/g)].map((m) => m[1].trim()));
  let links = '';
  fams.forEach((f) => [400, 500, 600, 700, 800].forEach((w) => { links += `<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=${f.replace(/ /g, '+')}:wght@${w}&display=block">`; links += `<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=${f.replace(/ /g, '+')}:ital,wght@1,${w}&display=block">`; }));
  fs.writeFileSync(dir + '/_hoja.html', html.replace('</head>', links + '</head>'));
  const b = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--allow-file-access-from-files'] });
  const p = await b.newPage(), errores = [];
  p.on('pageerror', (e) => errores.push('pageerror: ' + String(e.message).slice(0, 300)));
  p.on('console', (m) => { if (m.type() === 'error') errores.push('console: ' + m.text().slice(0, 300)); });
  await p.setViewport({ width: 1080, height: 1920, deviceScaleFactor: 0.25 });
  await p.goto('file://' + dir + '/_hoja.html', { waitUntil: 'networkidle0', timeout: 90000 });
  await p.evaluate(() => document.fonts.ready);
  await new Promise((r) => setTimeout(r, 1500));
  const TL = await p.evaluate(() => window.TL ? { fin: window.TL.fin, frases: window.TL.frases } : null);
  if (!TL) { errores.push('sin window.TL'); }
  const fr = Object.entries(TL ? TL.frases : {}).sort((a, b) => a[1][0] - b[1][0]);
  let ts = [[0.4, 'inicio']];
  fr.forEach(([f, [a, z]]) => ts.push([a + Math.min(1.4, Math.max(0.3, (z - a) * 0.6)), f]));
  if (TL) ts.push([TL.fin - 0.3, 'fin']);
  const seek = async (t) => p.evaluate((t) => {
    const tl = window.__timelines && window.__timelines.main; if (tl && tl.seek) tl.seek(t, false);
    window.dispatchEvent(new CustomEvent('hf-seek', { detail: { time: t } }));
  }, t);
  const tag = async (txt) => p.evaluate((txt) => {
    let d = document.getElementById('__tag'); if (!d) { d = document.createElement('div'); d.id = '__tag'; d.style.cssText = 'position:fixed;left:0;top:1856px;width:1080px;height:64px;background:#000;color:#ff0;font:700 48px monospace;z-index:99999;padding-left:16px'; document.body.appendChild(d); }
    d.textContent = txt; d.style.display = txt ? 'block' : 'none';
  }, txt);
  const chk = await p.evaluate(() => ({ tl: !!(window.__timelines && window.__timelines.main), w: document.documentElement.scrollWidth }));
  if (!chk.tl) errores.push('sin window.__timelines.main');
  for (let i = 0; i < ts.length; i++) {
    const [t, f] = ts[i]; await seek(t); await tag(`${t.toFixed(1)}s  ${f}`);
    await new Promise((r) => setTimeout(r, 60));
    await p.screenshot({ path: `cuadros/${String(i).padStart(3, '0')}.png` });
  }
  await tag('');
  const extra = (process.env.TIEMPOS || '').split(',').map((x) => parseFloat(x)).filter((x) => !isNaN(x));
  if (extra.length) {
    await p.setViewport({ width: 1080, height: 1920, deviceScaleFactor: 0.5 });
    for (const t of extra) { await seek(t); await new Promise((r) => setTimeout(r, 80)); await p.screenshot({ path: `salida/${process.env.CARPETA}-t${t.toFixed(1)}.png` }); }
  }
  fs.writeFileSync('salida/errores.txt', errores.join('\n') || 'sin errores');
  fs.writeFileSync('salida/info.txt', `dur ${TL ? TL.fin : '?'} s, ${ts.length} cuadros`);
  console.log(errores.length ? errores.join('\n') : 'sin errores'); console.log(`dur ${TL ? TL.fin : '?'}`);
  await b.close();
})().catch((e) => { console.error('fallo', e.message); fs.writeFileSync('salida/errores.txt', 'fallo: ' + e.message); process.exit(1); });
