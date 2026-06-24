// tests/test_update.js
//
// Pre-push smoke test: fires 100 queries at the backend /api/search endpoint
// and validates that each response is structurally sound — i.e. the returned
// cube can actually be displayed on the map.
//
// Also spot-checks /api/data for the first DATA_CHECK_LIMIT unique cubes
// found, confirming that province data actually comes back.
//
// Usage:
//   node tests/test_update.js [--base-url http://localhost:5000] [--delay 0]
//
// Exits with code 0 on full pass, 1 if any structural failures occurred.
//
// Run the backend server in a separate terminal before running this script:
//   node backend/server.js

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const SEARCH_RESULTS_PATH = path.join(__dirname, '..', 'classifier', 'input_data', 'searchResults.json');

const DEFAULT_BASE_URL    = 'http://localhost:5000';
const QUERY_COUNT         = 100;   // how many queries to benchmark
const DATA_CHECK_LIMIT    = 5;     // how many unique cubes to also test via /api/data

// Minimum provinces a cube must return for the map to be displayable.
const MIN_PROVINCE_COUNT = 8;

// ── Arg parsing ───────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const args = { baseUrl: DEFAULT_BASE_URL };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--base-url' && argv[i + 1]) args.baseUrl = argv[++i];
  }
  return args;
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────

async function post(url, body) {
  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { json = { _raw: text }; }
  return { status: res.status, body: json };
}

// ── Validation ────────────────────────────────────────────────────────────────

// Returns a list of failure strings, or [] if the search response is valid.
function validateSearchResponse(body) {
  const errs = [];

  if (!body.success)         errs.push('success !== true');
  if (!body.cubeId)          errs.push('missing cubeId');
  if (!body.title)           errs.push('missing title');
  if (typeof body.geoDimIndex !== 'number' || body.geoDimIndex < 0)
                             errs.push(`invalid geoDimIndex: ${body.geoDimIndex}`);
  if (!Array.isArray(body.provinces))
                             errs.push('provinces is not an array');
  else if (body.provinces.length < MIN_PROVINCE_COUNT)
                             errs.push(`only ${body.provinces.length} provinces (need >= ${MIN_PROVINCE_COUNT})`);
  else {
    const badProv = body.provinces.find(p => !p.name || typeof p.memberId !== 'number');
    if (badProv) errs.push(`malformed province entry: ${JSON.stringify(badProv)}`);
  }

  if (!Array.isArray(body.dimensionMeta))
                             errs.push('dimensionMeta is not an array');
  if (typeof body.tableUrl !== 'string' || !body.tableUrl.startsWith('https://'))
                             errs.push(`invalid tableUrl: ${body.tableUrl}`);
  if (typeof body.matchConfidence !== 'number' || body.matchConfidence < 0 || body.matchConfidence > 1)
                             errs.push(`matchConfidence out of range: ${body.matchConfidence}`);

  return errs;
}

// Returns a list of failure strings, or [] if the data response is valid.
function validateDataResponse(body, cubeId) {
  const errs = [];

  if (!body.success)
    errs.push(`[${cubeId}] /api/data success !== true`);
  if (!Array.isArray(body.provinces) || body.provinces.length < MIN_PROVINCE_COUNT)
    errs.push(`[${cubeId}] /api/data returned ${body.provinces?.length ?? 0} provinces`);
  else {
    const badProv = body.provinces.find(p => !p.province || p.value == null);
    if (badProv) errs.push(`[${cubeId}] malformed province data: ${JSON.stringify(badProv)}`);
  }

  return errs;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const { baseUrl } = parseArgs(process.argv.slice(2));

  console.log('='.repeat(60));
  console.log('  Canada Mapped — backend smoke test');
  console.log('='.repeat(60));
  console.log(`  Target:  ${baseUrl}`);
  console.log(`  Queries: ${QUERY_COUNT}`);
  console.log(`  /api/data spot-checks: first ${DATA_CHECK_LIMIT} unique cubes`);
  console.log('');

  // Verify server is reachable before doing anything else.
  try {
    const health = await fetch(`${baseUrl}/api/health`);
    const hBody  = await health.json();
    console.log(`  Health check: ${hBody.status ?? 'unknown'} (cubesLoaded=${hBody.cubesLoaded})`);
  } catch (err) {
    console.error(`\nERROR: Cannot reach ${baseUrl}/api/health — is the server running?\n  ${err.message}`);
    process.exit(1);
  }

  console.log('');

  // Load and sample queries.
  const rawResults = JSON.parse(await fs.readFile(SEARCH_RESULTS_PATH, 'utf8'));

  // Deduplicate by id and collect those with at least one non-null link
  // (same selection logic as test_result.js so we're testing realistic queries).
  const byId = new Map();
  for (const row of rawResults) {
    if (!byId.has(row.id)) byId.set(row.id, { id: row.id, query: row.query, hasLink: false });
    if (row.link) byId.get(row.id).hasLink = true;
  }

  const candidates = [...byId.values()].filter(q => q.hasLink);

  // Pick QUERY_COUNT queries spread evenly through the list for coverage across
  // all query types / categories rather than just the first N.
  const step     = Math.max(1, Math.floor(candidates.length / QUERY_COUNT));
  const selected = Array.from({ length: QUERY_COUNT }, (_, i) => candidates[i * step]).filter(Boolean);

  console.log(`  Loaded ${rawResults.length} rows → ${candidates.length} unique queries with links`);
  console.log(`  Selected ${selected.length} queries (every ${step}th)`);
  console.log('');

  // ── Phase 1: /api/search structural checks ────────────────────────────────

  console.log('Phase 1 — /api/search structural validation');
  console.log('-'.repeat(60));

  let searchPass    = 0;
  let searchNoResult = 0;
  let searchFail    = 0;
  const searchFailures = [];       // { query, errors }
  const cubesSeen  = new Map();    // cubeId → { body, defaultSelections }

  for (let i = 0; i < selected.length; i++) {
    const { query } = selected[i];
    let res;
    try {
      res = await post(`${baseUrl}/api/search`, { query, topK: 5 });
    } catch (err) {
      searchFail++;
      searchFailures.push({ query, errors: [`Network error: ${err.message}`] });
      console.log(`  [${i + 1}/${selected.length}] NETWORK ERR  "${query.slice(0, 55)}"`);
      continue;
    }

    if (res.status === 404) {
      searchNoResult++;
      console.log(`  [${i + 1}/${selected.length}] NO RESULT    "${query.slice(0, 55)}"`);
      continue;
    }

    if (res.status !== 200) {
      searchFail++;
      const msg = res.body?.error ?? `HTTP ${res.status}`;
      searchFailures.push({ query, errors: [msg] });
      console.log(`  [${i + 1}/${selected.length}] HTTP ${res.status}       "${query.slice(0, 55)}"`);
      continue;
    }

    const errs = validateSearchResponse(res.body);
    if (errs.length > 0) {
      searchFail++;
      searchFailures.push({ query, errors: errs });
      console.log(`  [${i + 1}/${selected.length}] INVALID      "${query.slice(0, 55)}"`);
      for (const e of errs) console.log(`               └ ${e}`);
      continue;
    }

    searchPass++;
    const conf = Math.round((res.body.matchConfidence ?? 0) * 100);
    console.log(`  [${i + 1}/${selected.length}] OK  ${conf}%  [${res.body.cubeId}] "${res.body.title.slice(0, 40)}"`);

    // Collect unique cubes for phase 2, building default selections.
    if (!cubesSeen.has(res.body.cubeId)) {
      const defaultSel = {};
      for (const dim of (res.body.dimensionMeta ?? [])) {
        const agg = dim.members?.find(m => m.isAggregate);
        const def = agg ?? dim.members?.[0];
        if (def) defaultSel[dim.dimIndex] = def.memberId;
      }
      cubesSeen.set(res.body.cubeId, {
        body: res.body,
        defaultSelections: defaultSel,
      });
    }
  }

  // ── Phase 2: /api/data spot-check ─────────────────────────────────────────

  console.log('');
  console.log(`Phase 2 — /api/data spot-check (first ${DATA_CHECK_LIMIT} unique cubes)`);
  console.log('-'.repeat(60));

  let dataPass = 0;
  let dataFail = 0;
  const dataFailures = [];

  const checkCubes = [...cubesSeen.entries()].slice(0, DATA_CHECK_LIMIT);

  for (const [cubeId, { body, defaultSelections }] of checkCubes) {
    let res;
    try {
      res = await post(`${baseUrl}/api/data`, {
        cubeId,
        geoDimIndex: body.geoDimIndex,
        provinces:   body.provinces,
        selections:  defaultSelections,
      });
    } catch (err) {
      dataFail++;
      dataFailures.push({ cubeId, errors: [`Network error: ${err.message}`] });
      console.log(`  [${cubeId}] NETWORK ERR  "${body.title.slice(0, 45)}"`);
      continue;
    }

    if (!res.body.success || res.status !== 200) {
      dataFail++;
      const msg = res.body?.error ?? `HTTP ${res.status}`;
      dataFailures.push({ cubeId, errors: [msg] });
      console.log(`  [${cubeId}] FAIL  "${body.title.slice(0, 45)}"  — ${msg}`);
      continue;
    }

    const errs = validateDataResponse(res.body, cubeId);
    if (errs.length > 0) {
      dataFail++;
      dataFailures.push({ cubeId, errors: errs });
      console.log(`  [${cubeId}] INVALID  "${body.title.slice(0, 45)}"`);
      for (const e of errs) console.log(`             └ ${e}`);
      continue;
    }

    dataPass++;
    const provCount = res.body.provinces.length;
    console.log(`  [${cubeId}] OK  ${provCount} provinces  "${body.title.slice(0, 45)}"`);
  }

  // ── Summary ───────────────────────────────────────────────────────────────

  console.log('');
  console.log('='.repeat(60));
  console.log('  RESULTS');
  console.log('='.repeat(60));
  console.log('');
  console.log(`  /api/search  (${selected.length} queries)`);
  console.log(`    Passed:     ${searchPass}`);
  console.log(`    No result:  ${searchNoResult}  (query may be unwinnable — not a bug)`);
  console.log(`    Failed:     ${searchFail}`);
  console.log('');
  console.log(`  /api/data  (${checkCubes.length} cubes spot-checked)`);
  console.log(`    Passed:     ${dataPass}`);
  console.log(`    Failed:     ${dataFail}`);

  if (searchFailures.length > 0) {
    console.log('');
    console.log('  /api/search failures:');
    for (const { query, errors } of searchFailures) {
      console.log(`    "${query.slice(0, 60)}"`);
      for (const e of errors) console.log(`      └ ${e}`);
    }
  }

  if (dataFailures.length > 0) {
    console.log('');
    console.log('  /api/data failures:');
    for (const { cubeId, errors } of dataFailures) {
      console.log(`    cubeId ${cubeId}`);
      for (const e of errors) console.log(`      └ ${e}`);
    }
  }

  console.log('');

  const totalFail = searchFail + dataFail;
  if (totalFail === 0) {
    console.log('  PASS — no structural failures detected.');
  } else {
    console.log(`  FAIL — ${totalFail} structural failure(s). Fix before pushing.`);
  }

  console.log('');
  process.exit(totalFail > 0 ? 1 : 0);
}

main().catch(err => {
  console.error('Test runner crashed:', err);
  process.exit(1);
});
