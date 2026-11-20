/* Small bar charts of CleanRUA benchmark results, drawn as plain SVG from a JSON file of episodes.
   Each <div class="rua-chart" data-src="..." data-chart="..."> becomes one chart with one bar per algorithm. */
(function () {
  const NAMES = { monitor: 'Monitor', delegate: 'Delegate', code: 'Code', direct: 'Direct', code_delegate: 'code_delegate' };
  const SVG = 'http://www.w3.org/2000/svg';
  const cache = {};

  const load = (src) => (cache[src] ||= fetch(src).then((response) => response.json()));
  const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
  const successes = (eps) => eps.filter((e) => e.success).length;
  const compact = (value) => value >= 1e6 ? `${(value / 1e6).toFixed(1)}M` : value >= 1e3 ? `${Math.round(value / 1e3)}k`
    : `${Math.round(value)}`;

  // value(episodes) -> number or null; label is printed on the bar tip, detail is shown on hover.
  const CHARTS = {
    success: { title: 'Success rate', top: 100, tick: (v) => `${v}%`,
      value: (eps) => 100 * successes(eps) / eps.length, label: (v) => `${Math.round(v)}%`,
      detail: (v, eps) => `${successes(eps)} of ${eps.length} episodes` },
    score: { title: 'Mean score', top: 100, tick: (v) => `${v}`,
      value: (eps) => mean(eps.map((e) => e.score)), label: (v) => v.toFixed(1),
      detail: (v, eps) => `over ${eps.length} episodes` },
    tokens: { title: 'Tokens per episode', tick: (v) => v >= 1e6 ? `${+(v / 1e6).toFixed(2)}M` : compact(v),
      value: (eps) => mean(eps.map((e) => e.tokens)), label: compact,
      detail: (v, eps) => `mean of ${eps.length} episodes` },
    tokens_per_success: { title: 'Tokens per success', tick: (v) => v >= 1e6 ? `${+(v / 1e6).toFixed(2)}M` : compact(v),
      value: (eps) => successes(eps) ? eps.reduce((a, e) => a + e.tokens, 0) / successes(eps) : null, label: compact,
      detail: (v, eps) => `${compact(eps.reduce((a, e) => a + e.tokens, 0))} tokens, ${successes(eps)} successes` },
  };

  function niceMax(value) {
    const power = 10 ** Math.floor(Math.log10(value));
    return [1, 2, 2.5, 5, 10].map((step) => step * power).find((step) => step >= value);
  }

  function el(name, attributes, parent) {
    const node = document.createElementNS(SVG, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    parent.appendChild(node);
    return node;
  }

  // A bar from the baseline with a 4 px rounded data-end.
  function barPath(x, y, width, height) {
    const r = Math.min(4, width / 2, height);
    return `M${x},${y + height}V${y + r}Q${x},${y} ${x + r},${y}H${x + width - r}Q${x + width},${y} ${x + width},${y + r}`
      + `V${y + height}Z`;
  }

  function render(root, data) {
    const spec = CHARTS[root.dataset.chart];
    const episodes = data.episodes.filter((e) => !e.invalid);
    const bars = data.algorithms.map((algorithm) => {
      const eps = episodes.filter((e) => e.algorithm === algorithm);
      return { algorithm, eps, value: eps.length ? spec.value(eps) : null };
    });
    const top = spec.top || niceMax(Math.max(...bars.map((b) => b.value || 0)) * 1.15);

    // Every chart has the same width, so all render at one scale; fewer bars get wider slots.
    const bar = 22, left = 34, width = 230, plotTop = 16, plotHeight = 120, height = plotTop + plotHeight + 18;
    const slot = (width - left - 4) / bars.length;
    const y = (value) => plotTop + plotHeight - (value / top) * plotHeight;

    root.textContent = '';
    const figure = document.createElement('div');
    figure.className = 'rua-figure';
    const title = document.createElement('div');
    title.className = 'rua-title';
    title.textContent = spec.title;
    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `${spec.title}: ` + bars.map((b) => `${NAMES[b.algorithm]} `
      + (b.value === null ? 'none' : spec.label(b.value))).join(', '));
    const tooltip = document.createElement('div');
    tooltip.className = 'rua-tooltip';
    tooltip.hidden = true;
    figure.append(title, svg, tooltip);
    root.appendChild(figure);

    for (let i = 0; i <= 2; i++) {  // Recessive hairline grid: baseline, middle and top.
      const value = (top / 2) * i;
      el('line', { x1: left, x2: width, y1: y(value), y2: y(value), class: i ? 'rua-grid' : 'rua-axis' }, svg);
      el('text', { x: left - 5, y: y(value) + 3.5, class: 'rua-tick', 'text-anchor': 'end' }, svg).textContent =
        spec.tick(value);
    }

    bars.forEach((b, i) => {
      const cx = left + slot * (i + 0.5);
      const hit = el('g', { class: 'rua-hit', tabindex: 0 }, svg);
      el('rect', { x: cx - slot / 2, y: plotTop, width: slot, height: plotHeight, fill: 'transparent' }, hit);
      const tip = b.value === null ? y(0) : y(b.value);
      if (b.value === null) {
        el('text', { x: cx, y: y(0) - 4, class: 'rua-tick', 'text-anchor': 'middle' }, svg).textContent = '–';
      } else {
        const h = Math.max(1, y(0) - tip);
        el('path', { d: barPath(cx - bar / 2, y(0) - h, bar, h), class: `rua-bar rua-${b.algorithm}` }, hit);
        el('text', { x: cx, y: y(0) - h - 4, class: 'rua-value', 'text-anchor': 'middle' }, svg).textContent =
          spec.label(b.value);
      }
      el('text', { x: cx, y: plotTop + plotHeight + 13, class: 'rua-category', 'text-anchor': 'middle' }, svg)
        .textContent = NAMES[b.algorithm];

      const show = () => {
        tooltip.textContent = '';
        const value = document.createElement('strong');
        value.textContent = b.value === null ? 'no success' : spec.label(b.value);
        const name = document.createElement('span');
        name.textContent = `${NAMES[b.algorithm]} · ${spec.detail(b.value, b.eps)}`;
        tooltip.append(value, name);
        tooltip.hidden = false;
        const scale = svg.getBoundingClientRect().width / width;
        tooltip.style.left = `${cx * scale}px`;
        tooltip.style.top = `${title.offsetHeight + (tip - 6) * scale}px`;
        hit.classList.add('rua-active');
      };
      const hide = () => { tooltip.hidden = true; hit.classList.remove('rua-active'); };
      hit.addEventListener('pointerenter', show);
      hit.addEventListener('focus', show);
      hit.addEventListener('pointerleave', hide);
      hit.addEventListener('blur', hide);
    });
  }

  function init() {
    for (const root of document.querySelectorAll('.rua-chart')) load(root.dataset.src).then((data) => render(root, data));
  }
  if (window.document$) window.document$.subscribe(init); else document.addEventListener('DOMContentLoaded', init);
})();
