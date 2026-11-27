/* Small bar charts of CleanRUA benchmark results, drawn as plain SVG from JSON files of episodes.
   Each <div class="rua-chart" data-src="a.json b.json" data-chart="..."> becomes one chart with a group of bars per
   algorithm and one bar per file: color is the algorithm, texture the file's model (the first solid, then hatched). */
(function () {
  const NAMES = { monitor: 'Monitor', delegate: 'Delegate', cap: 'CaP', direct: 'Direct', cap_delegate: 'cap_delegate' };
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

  let charts = 0;  // Pattern ids must be unique on the page.

  function render(root, sources) {
    const spec = CHARTS[root.dataset.chart], id = charts++;
    const groups = sources[0].algorithms.map((algorithm) => ({ algorithm, bars: sources.map((data, si) => {
      const eps = data.episodes.filter((e) => !e.invalid && e.algorithm === algorithm);
      return { model: data.model, hatch: si > 0, eps, value: eps.length ? spec.value(eps) : null };
    }) }));
    const values = groups.flatMap((g) => g.bars.map((b) => b.value || 0));
    const top = spec.top || niceMax(Math.max(...values) * 1.15);

    // Every chart has the same width and bar width, so all render at one scale; fewer groups get wider slots.
    const paired = sources.length > 1, bar = paired ? 18 : 22, gap = 2;
    const left = 34, width = 230, plotTop = 16, plotHeight = 120, height = plotTop + plotHeight + 18;
    const slot = (width - left - 4) / groups.length;
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
    svg.setAttribute('aria-label', `${spec.title}: ` + groups.flatMap((g) => g.bars.map((b) => `${NAMES[g.algorithm]}`
      + (paired ? ` ${b.model}` : '') + ` ${b.value === null ? 'none' : spec.label(b.value)}`)).join(', '));
    const tooltip = document.createElement('div');
    tooltip.className = 'rua-tooltip';
    tooltip.hidden = true;
    figure.append(title, svg, tooltip);
    root.appendChild(figure);

    // Hatched bars: the algorithm's color with surface-colored 45° stripes, defined inside this svg so that the
    // row's color variables apply.
    const defs = el('defs', {}, svg);
    for (const g of groups) {
      const pattern = el('pattern', { id: `rua-hatch-${id}-${g.algorithm}`, width: 4.2, height: 4.2,
                                      patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
      el('rect', { width: 4.2, height: 4.2, class: `rua-${g.algorithm}` }, pattern);
      el('line', { x1: 0, y1: 0, x2: 0, y2: 4.2, class: 'rua-stripe' }, pattern);
    }

    for (let i = 0; i <= 2; i++) {  // Recessive hairline grid: baseline, middle and top.
      const value = (top / 2) * i;
      el('line', { x1: left, x2: width, y1: y(value), y2: y(value), class: i ? 'rua-grid' : 'rua-axis' }, svg);
      el('text', { x: left - 5, y: y(value) + 3.5, class: 'rua-tick', 'text-anchor': 'end' }, svg).textContent =
        spec.tick(value);
    }

    groups.forEach((g, gi) => {
      const center = left + slot * (gi + 0.5), span = g.bars.length * bar + (g.bars.length - 1) * gap;
      g.bars.forEach((b, bi) => {
        const cx = center - span / 2 + bi * (bar + gap) + bar / 2;
        const hit = el('g', { class: 'rua-hit', tabindex: 0 }, svg);
        el('rect', { x: cx - (bar + gap) / 2, y: plotTop, width: bar + gap, height: plotHeight, fill: 'transparent' },
           hit);
        const tip = b.value === null ? y(0) : y(b.value);
        if (b.value === null) {
          el('text', { x: cx, y: y(0) - 4, class: 'rua-tick', 'text-anchor': 'middle' }, svg).textContent = '–';
        } else {
          const h = Math.max(1, y(0) - tip), d = barPath(cx - bar / 2, y(0) - h, bar, h);
          el('path', b.hatch ? { d, class: 'rua-bar', fill: `url(#rua-hatch-${id}-${g.algorithm})` }
            : { d, class: `rua-bar rua-${g.algorithm}` }, hit);
          el('text', { x: cx, y: y(0) - h - 4, class: paired ? 'rua-value rua-value-small' : 'rua-value',
                       'text-anchor': 'middle' }, svg).textContent = spec.label(b.value);
        }

        const show = () => {
          tooltip.textContent = '';
          const value = document.createElement('strong');
          value.textContent = b.value === null ? 'no success' : spec.label(b.value);
          const name = document.createElement('span');
          name.textContent = `${NAMES[g.algorithm]}${paired ? ` · ${b.model}` : ''} · ${spec.detail(b.value, b.eps)}`;
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
      el('text', { x: center, y: plotTop + plotHeight + 13, class: 'rua-category', 'text-anchor': 'middle' }, svg)
        .textContent = NAMES[g.algorithm];
    });
  }

  function init() {
    for (const root of document.querySelectorAll('.rua-chart')) Promise.all(root.dataset.src.split(/\s+/).map(load))
      .then((sources) => render(root, sources));
  }
  if (window.document$) window.document$.subscribe(init); else document.addEventListener('DOMContentLoaded', init);
})();
