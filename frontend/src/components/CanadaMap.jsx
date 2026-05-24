import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';

const CANADA_GEOJSON_URL =
  'https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/canada.geojson';

const ABBR = {
  'British Columbia': 'BC',      'Alberta': 'AB',          'Saskatchewan': 'SK',
  'Manitoba': 'MB',              'Ontario': 'ON',           'Quebec': 'QC',
  'New Brunswick': 'NB',         'Nova Scotia': 'NS',
  'Prince Edward Island': 'PEI', 'Newfoundland and Labrador': 'NL',
  'Yukon': 'YT',                 'Northwest Territories': 'NT', 'Nunavut': 'NU',
};

const BLUE_LOW  = '#dbeafe';
const BLUE_HIGH = '#1e3a8a';

const API_BASE = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL)
  ? import.meta.env.VITE_API_URL
  : 'http://localhost:5000';

async function apiFetch(path, body) {
  const res  = await fetch(`${API_BASE}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  });
  const text = await res.text();
  if (!text) throw new Error('Empty response from server');
  let json;
  try { json = JSON.parse(text); } catch { throw new Error(`Bad JSON: ${text.slice(0, 80)}`); }
  if (!res.ok || json.error) throw new Error(json.error ?? `Server error ${res.status}`);
  return json;
}

export default function CanadaMap() {
  const svgRef       = useRef(null);
  const containerRef = useRef(null);

  // Geo
  const [geoFeatures, setGeoFeatures] = useState(null);
  const [geoError,    setGeoError]    = useState(false);
  const [geoLoading,  setGeoLoading]  = useState(true);

  // Search phase
  const [query,       setQuery]       = useState('');
  const [searching,   setSearching]   = useState(false);
  const [searchError, setSearchError] = useState(null);

  // Cube metadata returned by /api/search
  // { cubeId, title, unit, tableUrl, geoDimIndex, provinces, dimensionMeta }
  const [cubeMeta, setCubeMeta] = useState(null);

  // Active selection: dimIndex (string) → memberId (number)
  // Initialised to the aggregate member of each dimension
  const [selections, setSelections] = useState({});

  // Data phase
  const [dataLoading, setDataLoading] = useState(false);
  const [dataError,   setDataError]   = useState(null);
  // [{ province, value, year }]
  const [provinceData, setProvinceData] = useState([]);
  const [dataYear,     setDataYear]     = useState('—');

  // UI
  const [tooltip,  setTooltip]  = useState(null);
  const [selected, setSelected] = useState(null);

  // value lookup for D3
  const valueMap = useMemo(
    () => Object.fromEntries(provinceData.map(p => [p.province, p.value])),
    [provinceData]
  );
  const nums   = Object.values(valueMap).filter(Number.isFinite);
  const minVal = nums.length ? Math.min(...nums) : 0;
  const maxVal = nums.length ? Math.max(...nums) : 1;

  // ── Load GeoJSON once ───────────────────────────────────────────────────────
  useEffect(() => {
    fetch(CANADA_GEOJSON_URL)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(raw => {
        const seen = new Set();
        const features = raw.features.filter(f => {
          const n = f.properties?.name;
          if (!n || seen.has(n)) return false;
          seen.add(n); return true;
        });
        setGeoFeatures(features);
        setGeoLoading(false);
      })
      .catch(() => { setGeoError(true); setGeoLoading(false); });
  }, []);

  // ── Step 1: Search — get cube metadata only ─────────────────────────────────
  async function handleSearch() {
    const q = query.trim();
    if (!q) return;
    setSearching(true);
    setSearchError(null);
    setDataError(null);
    setCubeMeta(null);
    setSelections({});
    setProvinceData([]);
    setSelected(null);
    try {
      const json = await apiFetch('/api/search', { query: q, topK: 5 });
      // Build default selections: pick aggregate member if present, else first
      const defaultSel = {};
      for (const dim of json.dimensionMeta) {
        const agg = dim.members.find(m => m.isAggregate);
        const def = agg ?? dim.members[0];
        if (def) defaultSel[dim.dimIndex] = def.memberId;
      }
      setCubeMeta(json);
      setSelections(defaultSel);
      // Immediately fetch data with defaults
      await fetchData(json, defaultSel);
    } catch (err) {
      setSearchError(err.message);
    } finally {
      setSearching(false);
    }
  }

  // ── Step 2: Fetch province data for current selections ──────────────────────
  const fetchData = useCallback(async (meta, sel) => {
    if (!meta) return;
    setDataLoading(true);
    setDataError(null);
    setSelected(null);
    try {
      const json = await apiFetch('/api/data', {
        cubeId:      meta.cubeId,
        geoDimIndex: meta.geoDimIndex,
        provinces:   meta.provinces,   // [{ name, memberId }]
        selections:  sel,              // { dimIndex: memberId }
      });
      setProvinceData(json.provinces);
      setDataYear(json.year);
    } catch (err) {
      setDataError(err.message);
      setProvinceData([]);
    } finally {
      setDataLoading(false);
    }
  }, []);

  // When user clicks a dimension button
  function handleMemberClick(dimIndex, memberId) {
    const newSel = { ...selections, [dimIndex]: memberId };
    setSelections(newSel);
    fetchData(cubeMeta, newSel);
  }

  // ── Draw map ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!geoFeatures || !svgRef.current || !containerRef.current) return;

    const W = containerRef.current.clientWidth  || 900;
    const H = containerRef.current.clientHeight || 520;

    const svg = d3.select(svgRef.current)
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('width', W).attr('height', H);
    svg.selectAll('*').remove();

    const geoCollection = { type: 'FeatureCollection', features: geoFeatures };
    const proj = d3.geoConicEqualArea()
      .parallels([49, 77]).rotate([96, 0]).center([0, 62])
      .fitExtent([[20, 20], [W - 20, H - 20]], geoCollection);
    const path = d3.geoPath().projection(proj);

    const colorScale = d3.scaleSequential()
      .domain([minVal, maxVal])
      .interpolator(d3.interpolateBlues);

    svg.append('g').selectAll('path')
      .data(geoFeatures).join('path')
        .attr('d', path)
        .attr('fill', d => {
          const v = valueMap[d.properties.name];
          return v != null ? colorScale(v) : '#d4d4d4';
        })
        .attr('stroke', '#1c1c1c').attr('stroke-width', 0.8).attr('stroke-linejoin', 'round')
        .style('cursor', 'pointer').style('transition', 'filter 0.15s ease')
        .on('mouseenter', function(event, d) {
          d3.select(this).attr('stroke-width', 2).attr('stroke', '#000').style('filter', 'brightness(1.1)');
          setTooltip({ name: d.properties.name, x: event.clientX, y: event.clientY });
        })
        .on('mousemove', function(event) {
          setTooltip(p => p ? { ...p, x: event.clientX, y: event.clientY } : null);
        })
        .on('mouseleave', function() {
          d3.select(this).attr('stroke-width', 0.8).attr('stroke', '#1c1c1c').style('filter', 'none');
          setTooltip(null);
        })
        .on('click', function(_, d) {
          const name   = d.properties.name;
          const record = provinceData.find(p => p.province === name);
          setSelected(record ? { name, value: record.value, year: record.year } : { name });
        });

    svg.append('g').style('pointer-events', 'none')
      .selectAll('text').data(geoFeatures).join('text')
        .attr('transform', d => { const [x,y] = path.centroid(d); return `translate(${x},${y})`; })
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('font-family', '"DM Mono","IBM Plex Mono",monospace')
        .attr('font-size', d =>
          ['Nunavut','Quebec','Ontario','British Columbia','Alberta'].includes(d.properties.name) ? '11px' : '9px')
        .attr('font-weight', '600').attr('fill', '#111')
        .attr('paint-order', 'stroke').attr('stroke', 'rgba(255,255,255,0.85)').attr('stroke-width', '2.5px')
        .text(d => ABBR[d.properties.name] || '');

  }, [geoFeatures, valueMap, minVal, maxVal]);

  // ── Formatting ───────────────────────────────────────────────────────────────
  const unit = cubeMeta?.unit ?? null;
  function fmtVal(v) {
    if (v == null) return '—';
    const u = (unit ?? '').toLowerCase();
    if (u.includes('dollar') || u.includes('cad')) return '$' + Math.round(v).toLocaleString();
    if (u.includes('percent'))                      return v + '%';
    return v.toLocaleString() + (unit ? ' ' + unit : '');
  }

  const metricName = cubeMeta?.title?.split(',')[0]?.trim() ?? '';
  const rankings   = [...provinceData].sort((a,b) => b.value - a.value);

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div style={S.page}>

      {/* Header */}
      <header style={S.header}>
        <div style={S.wordmark}>
          <span style={S.wPrimary}>STATCAN</span>
          <span style={S.wDiv}>/</span>
          <span style={S.wSecondary}>Explorer</span>
        </div>
        <div style={S.searchWrap}>
          <svg style={S.searchIcon} viewBox="0 0 20 20" fill="none">
            <circle cx="9" cy="9" r="6" stroke="#6b7280" strokeWidth="1.5"/>
            <path d="M14 14l3 3" stroke="#6b7280" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          <input
            style={S.searchInput}
            placeholder="Search a metric… e.g. provincial unemployment"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
          />
          {query && <button style={S.clearBtn} onClick={() => { setQuery(''); setSearchError(null); }}>✕</button>}
          <button style={{ ...S.searchBtn, opacity: (searching || dataLoading) ? 0.6 : 1 }}
            onClick={handleSearch} disabled={searching || dataLoading}>
            {searching ? '…' : 'Search'}
          </button>
        </div>
      </header>

      {/* Dimension filter bar */}
      {cubeMeta?.dimensionMeta?.length > 0 && (
        <div style={S.filterBar}>
          {dataLoading && <span style={S.fetchingBadge}>Fetching…</span>}
          {cubeMeta.dimensionMeta.map(dim => (
            <div key={dim.name} style={S.filterGroup}>
              <span style={S.filterLabel}>{dim.name}</span>
              <div style={S.filterBtns}>
                {dim.members.map(member => {
                  const active = selections[dim.dimIndex] === member.memberId;
                  return (
                    <button
                      key={member.memberId}
                      onClick={() => handleMemberClick(dim.dimIndex, member.memberId)}
                      disabled={dataLoading}
                      style={{
                        ...S.filterBtn,
                        ...(active      ? S.filterBtnActive   : {}),
                        ...(dataLoading ? S.filterBtnDisabled : {}),
                      }}
                      title={member.name}
                    >
                      {member.name}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Body */}
      <main style={S.body}>

        {/* Map */}
        <div ref={containerRef} style={S.mapArea}>
          {geoLoading && (
            <div style={S.loadState}>
              <div style={S.spinner} />
              <p style={{ color:'#6b7280', fontFamily:'monospace', marginTop:12 }}>Loading map…</p>
            </div>
          )}
          {geoError && <div style={S.loadState}><p style={{ color:'#dc2626', fontFamily:'monospace' }}>Failed to load map.</p></div>}
          {!geoLoading && !geoError && <svg ref={svgRef} style={{ display:'block', width:'100%', height:'100%' }} />}

          {/* Overlays */}
          {(searchError || dataError) && (
            <div style={S.errorBanner}>{searchError ?? dataError}</div>
          )}
          {!cubeMeta && !searching && !searchError && !geoLoading && (
            <div style={S.emptyPrompt}>Search a metric above to colour the map</div>
          )}
          {dataLoading && (
            <div style={S.fetchingOverlay}>
              <div style={S.spinnerSm} /> Fetching province data…
            </div>
          )}
        </div>

        {/* Sidebar */}
        <aside style={S.sidebar}>
          <div style={S.metaCard}>
            <div style={S.metaLabel}>Metric</div>
            <div style={S.metaValue}>{metricName || '—'}</div>
            <div style={{ ...S.metaLabel, marginTop:8 }}>Year</div>
            <div style={S.metaValue}>{dataYear}</div>
            <div style={{ ...S.metaLabel, marginTop:8 }}>Unit</div>
            <div style={S.metaValue}>{unit ?? 'N/A'}</div>
            {cubeMeta?.tableUrl && (
              <a href={cubeMeta.tableUrl} target="_blank" rel="noreferrer" style={S.tableLink}>
                View source table ↗
              </a>
            )}
          </div>

          <div style={S.legendCard}>
            <div style={S.legendTitle}>Legend</div>
            <div style={{ height:12, borderRadius:6, background:`linear-gradient(to right, ${BLUE_LOW}, ${BLUE_HIGH})`, margin:'8px 0 6px', opacity: rankings.length ? 1 : 0.3 }} />
            <div style={S.legendLabels}>
              <span>{rankings.length ? fmtVal(rankings.at(-1).value) : '—'}</span>
              <span>{rankings.length ? fmtVal(rankings[0].value)     : '—'}</span>
            </div>
            <div style={S.legendNA}><span style={S.naBox} /><span>No data</span></div>
          </div>

          {selected ? (
            <div style={S.detailCard}>
              <div style={S.detailName}>{selected.name}</div>
              {selected.value != null
                ? <><div style={S.detailValue}>{fmtVal(selected.value)}</div><div style={S.detailMeta}>{metricName} · {selected.year ?? dataYear}</div></>
                : <div style={S.detailMeta}>No data for this combination</div>
              }
              <button style={S.closeBtn} onClick={() => setSelected(null)}>✕</button>
            </div>
          ) : (
            <div style={S.hintCard}>Click a province to see details</div>
          )}

          <div style={S.rankCard}>
            <div style={S.rankTitle}>Rankings</div>
            {rankings.length > 0
              ? rankings.map((p, i) => (
                  <div key={p.province}
                    style={{ ...S.rankRow, background: selected?.name === p.province ? '#eff6ff' : 'transparent', cursor:'pointer' }}
                    onClick={() => setSelected({ name:p.province, value:p.value, year:p.year })}>
                    <span style={S.rankNum}>{i + 1}</span>
                    <span style={S.rankName}>{ABBR[p.province] ?? p.province}</span>
                    <span style={S.rankVal}>{fmtVal(p.value)}</span>
                  </div>
                ))
              : <div style={{ fontSize:11, color:'#d1d5db', fontStyle:'italic' }}>
                  {cubeMeta ? 'No data for selection' : 'No data yet'}
                </div>
            }
          </div>
        </aside>
      </main>

      {tooltip && (
        <div style={{ ...S.tooltip, left: tooltip.x + 14, top: tooltip.y - 48 }}>
          <div style={S.ttName}>{tooltip.name}</div>
          <div style={S.ttVal}>
            {valueMap[tooltip.name] != null ? fmtVal(valueMap[tooltip.name]) : 'No data'}
          </div>
        </div>
      )}
    </div>
  );
}

const S = {
  page:        { display:'flex', flexDirection:'column', height:'100vh', width:'100vw', background:'#f5f4f0', fontFamily:'"DM Mono","IBM Plex Mono","Courier New",monospace', overflow:'hidden' },
  header:      { display:'flex', alignItems:'center', gap:20, padding:'10px 20px', background:'#111', borderBottom:'1px solid #333', flexShrink:0, flexWrap:'wrap' },
  wordmark:    { display:'flex', alignItems:'baseline', gap:6 },
  wPrimary:    { color:'#fff', fontWeight:'700', fontSize:18, letterSpacing:'0.08em' },
  wDiv:        { color:'#555', fontSize:18 },
  wSecondary:  { color:'#9ca3af', fontWeight:'400', fontSize:14 },
  searchWrap:  { flex:1, minWidth:220, maxWidth:520, display:'flex', alignItems:'center', background:'#1f1f1f', border:'1px solid #333', borderRadius:6, padding:'0 4px 0 10px', gap:4 },
  searchIcon:  { width:16, height:16, flexShrink:0 },
  searchInput: { flex:1, background:'transparent', border:'none', outline:'none', color:'#e5e7eb', fontFamily:'inherit', fontSize:13, padding:'8px 6px' },
  clearBtn:    { background:'none', border:'none', color:'#6b7280', cursor:'pointer', fontSize:12, padding:'0 4px' },
  searchBtn:   { background:'#1d4ed8', color:'#fff', border:'none', borderRadius:4, fontFamily:'inherit', fontSize:12, fontWeight:'600', padding:'5px 12px', cursor:'pointer', flexShrink:0 },

  filterBar:       { display:'flex', flexWrap:'wrap', alignItems:'flex-start', gap:12, padding:'10px 20px', background:'#1a1a1a', borderBottom:'1px solid #2a2a2a', flexShrink:0 },
  fetchingBadge:   { fontSize:10, color:'#6b7280', alignSelf:'center', fontStyle:'italic' },
  filterGroup:     { display:'flex', flexDirection:'column', gap:5 },
  filterLabel:     { fontSize:10, color:'#6b7280', textTransform:'uppercase', letterSpacing:'0.08em' },
  filterBtns:      { display:'flex', gap:4, flexWrap:'wrap' },
  filterBtn:       { background:'#2a2a2a', border:'1px solid #3a3a3a', color:'#9ca3af', borderRadius:4, fontFamily:'inherit', fontSize:11, padding:'3px 10px', cursor:'pointer', transition:'all 0.1s', whiteSpace:'nowrap' },
  filterBtnActive: { background:'#1d4ed8', border:'1px solid #1d4ed8', color:'#fff' },
  filterBtnDisabled:{ opacity:0.4, cursor:'not-allowed' },

  body:           { display:'flex', flex:1, overflow:'hidden' },
  mapArea:        { flex:1, position:'relative', background:'#e8e6e0', display:'flex', alignItems:'center', justifyContent:'center' },
  loadState:      { display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center' },
  spinner:        { width:32, height:32, borderRadius:'50%', border:'3px solid #d1d5db', borderTopColor:'#111', animation:'spin 0.7s linear infinite' },
  spinnerSm:      { width:14, height:14, borderRadius:'50%', border:'2px solid #9ca3af', borderTopColor:'#1d4ed8', animation:'spin 0.7s linear infinite', display:'inline-block', marginRight:6 },
  fetchingOverlay:{ position:'absolute', bottom:20, left:'50%', transform:'translateX(-50%)', background:'rgba(255,255,255,0.92)', borderRadius:6, padding:'8px 16px', fontSize:12, color:'#374151', display:'flex', alignItems:'center', backdropFilter:'blur(4px)' },
  errorBanner:    { position:'absolute', bottom:20, left:'50%', transform:'translateX(-50%)', background:'#fef2f2', border:'1px solid #fca5a5', color:'#dc2626', borderRadius:6, padding:'8px 16px', fontSize:12, maxWidth:'80%' },
  emptyPrompt:    { position:'absolute', bottom:20, left:'50%', transform:'translateX(-50%)', background:'rgba(255,255,255,0.85)', borderRadius:6, padding:'8px 16px', fontSize:12, color:'#6b7280', whiteSpace:'nowrap', backdropFilter:'blur(4px)' },

  sidebar:     { width:220, flexShrink:0, background:'#fff', borderLeft:'1px solid #e5e7eb', overflowY:'auto', display:'flex', flexDirection:'column' },
  metaCard:    { padding:'16px 16px 12px', borderBottom:'1px solid #f0f0f0' },
  metaLabel:   { fontSize:10, color:'#9ca3af', textTransform:'uppercase', letterSpacing:'0.08em' },
  metaValue:   { fontSize:13, color:'#111', fontWeight:'600', marginTop:2 },
  tableLink:   { display:'inline-block', marginTop:10, fontSize:11, color:'#1d4ed8', textDecoration:'none', borderBottom:'1px solid #bfdbfe', paddingBottom:1 },
  legendCard:  { padding:'14px 16px', borderBottom:'1px solid #f0f0f0' },
  legendTitle: { fontSize:10, color:'#9ca3af', textTransform:'uppercase', letterSpacing:'0.08em' },
  legendLabels:{ display:'flex', justifyContent:'space-between', fontSize:10, color:'#6b7280' },
  legendNA:    { display:'flex', alignItems:'center', gap:6, fontSize:10, color:'#6b7280', marginTop:6 },
  naBox:       { display:'inline-block', width:12, height:12, background:'#d4d4d4', borderRadius:2 },
  detailCard:  { padding:'14px 16px', borderBottom:'1px solid #f0f0f0', position:'relative', background:'#fafafa' },
  detailName:  { fontSize:12, fontWeight:'700', color:'#111', marginBottom:4 },
  detailValue: { fontSize:22, fontWeight:'700', color:'#1d4ed8', lineHeight:1.1 },
  detailMeta:  { fontSize:10, color:'#9ca3af', marginTop:4 },
  closeBtn:    { position:'absolute', top:10, right:10, background:'none', border:'none', color:'#9ca3af', cursor:'pointer', fontSize:12 },
  hintCard:    { padding:'14px 16px', borderBottom:'1px solid #f0f0f0', fontSize:11, color:'#9ca3af', fontStyle:'italic' },
  rankCard:    { flex:1, padding:'14px 16px', overflowY:'auto' },
  rankTitle:   { fontSize:10, color:'#9ca3af', textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:8 },
  rankRow:     { display:'flex', alignItems:'center', gap:8, padding:'5px 4px', borderRadius:4, transition:'background 0.1s' },
  rankNum:     { fontSize:10, color:'#d1d5db', width:16, textAlign:'right', flexShrink:0 },
  rankName:    { fontSize:11, color:'#374151', flex:1 },
  rankVal:     { fontSize:11, color:'#111', fontWeight:'600' },
  tooltip:     { position:'fixed', zIndex:1000, pointerEvents:'none', background:'#111', color:'#fff', borderRadius:6, padding:'8px 12px', boxShadow:'0 4px 16px rgba(0,0,0,0.25)' },
  ttName:      { fontSize:11, color:'#9ca3af', marginBottom:2 },
  ttVal:       { fontSize:14, fontWeight:'700' },
};
