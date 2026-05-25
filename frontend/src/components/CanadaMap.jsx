import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import './CanadaMap.css';

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

const API_BASE = import.meta.env.VITE_API_URL ?? '';

async function apiFetch(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
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

function DimensionFilter({ dim, selectedId, onChange, disabled }) {
  return (
    <div className="dim-filter">
      <label className="dim-label">{dim.name}</label>
      <select
        className="dim-select"
        style={{ opacity: disabled ? 0.5 : 1 }}
        value={selectedId ?? ''}
        disabled={disabled}
        onChange={e => onChange(dim.dimIndex, Number(e.target.value))}
      >
        {dim.members.map(m => (
          <option key={m.memberId} value={m.memberId}>{m.name}</option>
        ))}
      </select>
    </div>
  );
}

export default function CanadaMap() {
  const svgRef       = useRef(null);
  const containerRef = useRef(null);

  // Geo
  const [geoFeatures, setGeoFeatures] = useState(null);
  const [geoError,    setGeoError]    = useState(false);
  const [geoLoading,  setGeoLoading]  = useState(true);

  // Search
  const [query,       setQuery]       = useState('');
  const [searching,   setSearching]   = useState(false);
  const [searchError, setSearchError] = useState(null);

  // Cube metadata
  const [cubeMeta, setCubeMeta] = useState(null);

  // Active selections: dimIndex → memberId
  const [selections, setSelections] = useState({});

  // Data
  const [dataLoading,  setDataLoading]  = useState(false);
  const [dataError,    setDataError]    = useState(null);
  const [provinceData, setProvinceData] = useState([]);
  const [dataYear,     setDataYear]     = useState('—');

  // UI
  const [tooltip,  setTooltip]  = useState(null);
  const [selected, setSelected] = useState(null);

  const valueMap = useMemo(
    () => Object.fromEntries(provinceData.map(p => [p.province, p.value])),
    [provinceData]
  );
  const nums   = Object.values(valueMap).filter(Number.isFinite);
  const minVal = nums.length ? Math.min(...nums) : 0;
  const maxVal = nums.length ? Math.max(...nums) : 1;

  // ── GeoJSON ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    fetch(CANADA_GEOJSON_URL)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(raw => {
        const seen = new Set();
        setGeoFeatures(raw.features.filter(f => {
          const n = f.properties?.name;
          if (!n || seen.has(n)) return false;
          seen.add(n); return true;
        }));
        setGeoLoading(false);
      })
      .catch(() => { setGeoError(true); setGeoLoading(false); });
  }, []);

  // ── Fetch data for current selections ────────────────────────────────────────
  const fetchData = useCallback(async (meta, sel) => {
    if (!meta) return;
    setDataLoading(true);
    setDataError(null);
    setSelected(null);
    try {
      const json = await apiFetch(`/${API_BASE}/data`, {
        cubeId:      meta.cubeId,
        geoDimIndex: meta.geoDimIndex,
        provinces:   meta.provinces,
        selections:  sel,
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

  // ── Search ───────────────────────────────────────────────────────────────────
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
      const json = await apiFetch(`/${API_BASE}/search`, { query: q, topK: 5 });
      const defaultSel = {};
      for (const dim of json.dimensionMeta) {
        const agg = dim.members.find(m => m.isAggregate);
        const def = agg ?? dim.members[0];
        if (def) defaultSel[dim.dimIndex] = def.memberId;
      }
      setCubeMeta(json);
      setSelections(defaultSel);
      await fetchData(json, defaultSel);
    } catch (err) {
      setSearchError(err.message);
    } finally {
      setSearching(false);
    }
  }

  function handleDimChange(dimIndex, memberId) {
    setSelections(prev => ({ ...prev, [dimIndex]: memberId }));
  }

  function handleApplyFilters() {
    fetchData(cubeMeta, selections);
  }

  // ── D3 map ───────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!geoFeatures || !svgRef.current || !containerRef.current) return;

    const W = containerRef.current.clientWidth  || 900;
    const H = containerRef.current.clientHeight || 520;

    const svg = d3.select(svgRef.current)
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('width', W).attr('height', H);
    svg.selectAll('*').remove();

    const geoCol = { type: 'FeatureCollection', features: geoFeatures };
    const proj   = d3.geoConicEqualArea()
      .parallels([49, 77]).rotate([96, 0]).center([0, 62])
      .fitExtent([[20, 20], [W - 20, H - 20]], geoCol);
    const path = d3.geoPath().projection(proj);

    const colorScale = d3.scaleSequential()
      .domain([minVal, maxVal])
      .interpolator(d3.interpolateBlues);

    svg.append('g').selectAll('path')
      .data(geoFeatures).join('path')
        .attr('d', path)
        .attr('fill', d => { const v = valueMap[d.properties.name]; return v != null ? colorScale(v) : '#d4d4d4'; })
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
  const unit       = cubeMeta?.unit ?? null;
  const metricName = cubeMeta?.title?.split(',')[0]?.trim() ?? '';
  const rankings   = [...provinceData].sort((a,b) => b.value - a.value);

  function fmtVal(v) {
    if (v == null) return '—';
    const u = (unit ?? '').toLowerCase();
    if (u.includes('dollar') || u.includes('cad')) return '$' + Math.round(v).toLocaleString();
    if (u.includes('percent'))                      return v + '%';
    return v.toLocaleString() + (unit ? ' ' + unit : '');
  }

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="page">

      {/* Header */}
      <header className="header">
        <div className="wordmark">
          <span className="w-primary">STATCAN</span>
          <span className="w-div">/</span>
          <span className="w-secondary">Explorer</span>
        </div>
        <div className="search-wrap">
          <svg className="search-icon" viewBox="0 0 20 20" fill="none">
            <circle cx="9" cy="9" r="6" stroke="#6b7280" strokeWidth="1.5"/>
            <path d="M14 14l3 3" stroke="#6b7280" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          <input
            className="search-input"
            placeholder="Search a metric… e.g. provincial unemployment"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
          />
          {query && (
            <button className="clear-btn" onClick={() => { setQuery(''); setSearchError(null); }}>✕</button>
          )}
          <button
            className="search-btn"
            style={{ opacity: (searching || dataLoading) ? 0.6 : 1 }}
            onClick={handleSearch}
            disabled={searching || dataLoading}
          >
            {searching ? '…' : 'Search'}
          </button>
        </div>
      </header>

      {/* Body */}
      <main className="body">

        {/* Map area */}
        <div ref={containerRef} className="map-area">
          {geoLoading && (
            <div className="load-state">
              <div className="spinner" />
              <p style={{ color:'#6b7280', fontFamily:'monospace', marginTop:12 }}>Loading map…</p>
            </div>
          )}
          {geoError && (
            <div className="load-state">
              <p style={{ color:'#dc2626', fontFamily:'monospace' }}>Failed to load map boundaries.</p>
            </div>
          )}
          {!geoLoading && !geoError && (
            <svg ref={svgRef} style={{ display:'block', width:'100%', height:'100%' }} />
          )}

          {(searchError || dataError) && (
            <div className="error-banner">{searchError ?? dataError}</div>
          )}
          {!cubeMeta && !searching && !searchError && !geoLoading && (
            <div className="empty-prompt">Search a metric above to colour the map</div>
          )}
          {dataLoading && (
            <div className="fetching-overlay">
              <div className="spinner-sm" />Fetching province data…
            </div>
          )}
        </div>

        {/* Sidebar with internal scrolling layout enabled */}
        <aside className="sidebar">

          {/* Metric info */}
          <div className="side-section">
            <div className="meta-label">Metric</div>
            <div className="meta-value">{metricName || '—'}</div>
            <div className="meta-label" style={{ marginTop: 8 }}>Year</div>
            <div className="meta-value">{dataYear}</div>
            <div className="meta-label" style={{ marginTop: 8 }}>Unit</div>
            <div className="meta-value">{unit ?? 'N/A'}</div>
            {cubeMeta?.tableUrl && (
              <a href={cubeMeta.tableUrl} target="_blank" rel="noreferrer" className="table-link">
                View source table ↗
              </a>
            )}
          </div>

          {/* Dimension filters */}
          {cubeMeta?.dimensionMeta?.length > 0 && (
            <div className="filter-section">
              <div className="filter-header">
                <span className="section-title">Filters</span>
                {dataLoading && <span className="fetching-badge">updating…</span>}
              </div>
              <div className="filter-scroll">
                {cubeMeta.dimensionMeta.map(dim => (
                  <DimensionFilter
                    key={dim.dimIndex}
                    dim={dim}
                    selectedId={selections[dim.dimIndex]}
                    onChange={handleDimChange}
                    disabled={dataLoading}
                  />
                ))}
                <button 
                  className="filter-action-btn"
                  onClick={handleApplyFilters}
                  disabled={dataLoading}
                  style={{ opacity: dataLoading ? 0.6 : 1 }}
                >
                  Filter
                </button>
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="side-section">
            <div className="section-title">Legend</div>
            <div style={{
              height: 12, borderRadius: 6,
              background: `linear-gradient(to right, ${BLUE_LOW}, ${BLUE_HIGH})`,
              margin: '8px 0 6px', opacity: rankings.length ? 1 : 0.3,
            }} />
            <div className="legend-labels">
              <span>{rankings.length ? fmtVal(rankings.at(-1).value) : '—'}</span>
              <span>{rankings.length ? fmtVal(rankings[0].value)     : '—'}</span>
            </div>
            <div className="legend-na"><span className="na-box"/><span>No data</span></div>
          </div>

          {/* Province detail card */}
          {selected ? (
            <div className="detail-card">
              <div className="detail-name">{selected.name}</div>
              {selected.value != null
                ? <>
                    <div className="detail-value">{fmtVal(selected.value)}</div>
                    <div className="detail-meta">{metricName} · {selected.year ?? dataYear}</div>
                  </>
                : <div className="detail-meta">No data for this combination</div>
              }
              <button className="close-btn" onClick={() => setSelected(null)}>✕</button>
            </div>
          ) : (
            <div className="hint-card">Click a province to see details</div>
          )}

          {/* Rankings */}
          <div className="rank-card">
            <div className="section-title">Rankings</div>
            {rankings.length > 0
              ? rankings.map((p, i) => (
                  <div key={p.province}
                    className="rank-row"
                    style={{
                      background: selected?.name === p.province ? '#eff6ff' : 'transparent',
                    }}
                    onClick={() => setSelected({ name:p.province, value:p.value, year:p.year })}
                  >
                    <span className="rank-num">{i + 1}</span>
                    <span className="rank-name">{ABBR[p.province] ?? p.province}</span>
                    <span className="rank-val">{fmtVal(p.value)}</span>
                  </div>
                ))
              : <div style={{ fontSize:11, color:'#d1d5db', fontStyle:'italic', marginTop:4 }}>
                  {cubeMeta ? 'No data for this selection' : 'No data yet'}
                </div>
            }
          </div>
        </aside>
      </main>

      {/* Tooltip */}
      {tooltip && (
        <div className="tooltip" style={{ left: tooltip.x + 14, top: tooltip.y - 48 }}>
          <div className="tt-name">{tooltip.name}</div>
          <div className="tt-val">
            {valueMap[tooltip.name] != null ? fmtVal(valueMap[tooltip.name]) : 'No data'}
          </div>
        </div>
      )}
    </div>
  );
}