// classifier/test_result.js
//
// Benchmarks server.js's /api/search endpoint against ground-truth StatCan
// table links (searchResults.json) collected from Google/Serper search
// results for the same queries.
//
// HOW MATCHING WORKS
// -------------------
// searchResults.json contains one row per (query, candidate link) pair, so
// the same query id can appear multiple times with different links — these
// are multiple acceptable answers for that query, not duplicates to dedupe
// away. A query is scored as a HIT if server.js's returned cubeId matches
// ANY of that query's expected cubeIds, not just the first one.
//
// Each link's pid (e.g. ...?pid=1410033502) encodes an 8-digit StatCan
// cubeId plus a 2-digit table-version suffix (mirrors the padding logic in
// server.js's own tableUrl construction: cubeId.padStart(8,'0') + '01',
// unless the cubeId already ends in '01'). Only the 8-digit cubeId is
// compared — the suffix is a StatCan table-version detail server.js doesn't
// choose, so comparing on the full 10-digit pid would produce false
// negatives unrelated to whether search actually found the right cube.
//
// QUERY BUCKETS
// -------------------
// Every distinct query (by id) falls into exactly one bucket:
//   - NO_LINK:       every row for this id has link === null. No ground
//                     truth exists; excluded from accuracy entirely.
//   - UNWINNABLE:     has at least one link, but none of its expected
//                     cubeIds exist in cubesMetadata.json. server.js can
//                     never return a cube that isn't in its own search
//                     space, so these can't be "missed" in any meaningful
//                     sense; excluded from the accuracy denominator and
//                     reported separately.
//   - WINNABLE:       has at least one expected cubeId that DOES exist in
//                     cubesMetadata.json. These make up the accuracy
//                     denominator — every other query type would inflate or
//                     deflate accuracy for reasons that have nothing to do
//                     with search quality.
//
// CATEGORY ATTRIBUTION
// -------------------
// Per-category accuracy uses the EXPECTED cube's category (via
// subjectCodeToCategory.json, the same source Search_Utils.js's
// rerankByCategory uses) — specifically, the first winnable expected
// cubeId's category for that query. This was chosen over using the MLP's
// live category prediction for the query text, since that would conflate
// two different questions ("did search find the right cube" vs "did the
// classifier predict the right category for this query") into one number.
//
// OUTPUTS
// -------------------
//   search_Serper_result.txt — overall accuracy, per-category accuracy
//                               table, and bucket counts (no-link /
//                               unwinnable / winnable / hit / miss).
//   mismatched.json           — one entry per WINNABLE query that missed:
//                               { id, query, expected: [...], actual }.
//
// USAGE
// -------------------
//   node test_results.js [--base-url http://localhost:5000] [--delay 60]
//
// Requires server.js to already be running and reachable at --base-url
// (default http://localhost:5000) — this script makes real HTTP requests
// to /api/search for every winnable+unwinnable... actually every query with
// at least one link (so unwinnable queries are also confirmed as misses by
// the live server, not just assumed), one request per distinct query id.
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

// 1. Define __filename and __dirname first
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 2. Now you can safely use __dirname
const PROJECT_ROOT = path.join(__dirname, "..");

const SEARCH_RESULTS_PATH = path.join(PROJECT_ROOT, "classifier", "input_data", "searchResults.json");
const CUBES_METADATA_PATH = path.join(
  __dirname,
  "..",
  "canada-data-pipeline",
  "src",
  "collectors",
  "cubesMetadata.json"
);

const SUBJECT_CODE_MAP_PATH = path.join(
  __dirname,
  "..",
  "canada-data-pipeline",
  "src",
  "collectors",
  "subjectCodeToCategory.json"
);

const TXT_OUTPUT_PATH = path.join(__dirname, "search_Serper_result.txt");
const MISMATCHED_OUTPUT_PATH = path.join(__dirname, "mismatched.json");

const PID_PATTERN = /pid=(\d+)/;

// Small delay between requests so this doesn't hammer the live StatCan API
// (server.js's own /api/data handler already rate-limits at 60ms between
// per-province calls; this is the equivalent courtesy at the query level).
const DEFAULT_DELAY_MS = 60;
const DEFAULT_BASE_URL = "http://localhost:5000";

function parseArgs(argv) {
  const args = { baseUrl: DEFAULT_BASE_URL, delayMs: DEFAULT_DELAY_MS };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--base-url" && argv[i + 1]) {
      args.baseUrl = argv[++i];
    } else if (argv[i] === "--delay" && argv[i + 1]) {
      args.delayMs = Number(argv[++i]);
    }
  }
  return args;
}

async function loadJson(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  return JSON.parse(raw);
}

// Extracts the 8-digit cubeId from a StatCan table URL's pid query param.
// Returns null if the link is missing or doesn't contain a pid.
function cubeIdFromLink(link) {
  if (!link) return null;
  const match = PID_PATTERN.exec(link);
  if (!match) return null;
  return match[1].slice(0, -2); // strip the 2-digit table-version suffix
}

// Groups searchResults.json's flat rows by query id, since the same id can
// have multiple acceptable links.
function groupByQueryId(rows) {
  const byId = new Map();
  for (const row of rows) {
    if (!byId.has(row.id)) {
      byId.set(row.id, { id: row.id, query: row.query, links: [] });
    }
    if (row.link) byId.get(row.id).links.push(row.link);
  }
  return [...byId.values()];
}

function categoryForCubeId(cubeId, cubesById, subjectCodeMap) {
  const cube = cubesById.get(cubeId);
  if (!cube?.subjectCode?.length) return null;
  for (const code of cube.subjectCode) {
    const category = subjectCodeMap[code];
    if (category) return category;
  }
  return null;
}

async function callSearch(baseUrl, query) {
  const res = await fetch(`${baseUrl}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, topK: 5 })
  });

  if (res.status === 404) {
    return { cubeId: null, category: null, raw: await res.json() };
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`/api/search returned ${res.status}: ${text.slice(0, 200)}`);
  }

  const body = await res.json();
  return { cubeId: body.cubeId ?? null, category: body.cubeCategory ?? null, raw: body };
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function formatPercent(hit, total) {
  if (total === 0) return "n/a (0 queries)";
  return `${((hit / total) * 100).toFixed(1)}% (${hit}/${total})`;
}

async function main() {
  const { baseUrl, delayMs } = parseArgs(process.argv.slice(2));

  console.log(`Target server: ${baseUrl}`);
  console.log("Loading searchResults.json, cubesMetadata.json, subjectCodeToCategory.json…");

  const [rawResults, cubes, subjectCodeMap] = await Promise.all([
    loadJson(SEARCH_RESULTS_PATH),
    loadJson(CUBES_METADATA_PATH),
    loadJson(SUBJECT_CODE_MAP_PATH)
  ]);

  const cubesById = new Map(cubes.map(c => [c.cubeId, c]));
  const queries = groupByQueryId(rawResults);

  console.log(`Distinct queries: ${queries.length}`);

  // Bucket every query and compute its set of expected (winnable) cubeIds
  // up front, before making any HTTP calls.
  const noLinkQueries = [];
  const unwinnableQueries = [];
  const winnableQueries = []; // { id, query, expectedCubeIds: Set, category }

  for (const q of queries) {
    if (q.links.length === 0) {
      noLinkQueries.push(q);
      continue;
    }

    const expectedCubeIds = new Set(
      q.links.map(cubeIdFromLink).filter(Boolean)
    );
    const winnableCubeIds = [...expectedCubeIds].filter(id => cubesById.has(id));

    if (winnableCubeIds.length === 0) {
      unwinnableQueries.push(q);
      continue;
    }

    const category = categoryForCubeId(winnableCubeIds[0], cubesById, subjectCodeMap);
    winnableQueries.push({
      id: q.id,
      query: q.query,
      expectedCubeIds: new Set(winnableCubeIds),
      category
    });
  }

  console.log(`  No-link (no ground truth):        ${noLinkQueries.length}`);
  console.log(`  Unwinnable (expected cube absent): ${unwinnableQueries.length}`);
  console.log(`  Winnable (in accuracy denominator):${winnableQueries.length}`);
  console.log(`\nQuerying ${baseUrl}/api/search for ${winnableQueries.length} winnable queries…\n`);

  let hitCount = 0;
  const mismatched = [];

  // category -> { hit, total }
  const categoryStats = new Map();

  for (let i = 0; i < winnableQueries.length; i++) {
    const q = winnableQueries[i];

    let result;
    try {
      result = await callSearch(baseUrl, q.query);
    } catch (err) {
      console.error(`[${i + 1}/${winnableQueries.length}] ERROR for "${q.query}": ${err.message}`);
      result = { cubeId: null, category: null, raw: { error: err.message } };
    }

    const isHit = result.cubeId !== null && q.expectedCubeIds.has(result.cubeId);
    if (isHit) hitCount++;

    const categoryKey = q.category ?? "Unknown";
    if (!categoryStats.has(categoryKey)) categoryStats.set(categoryKey, { hit: 0, total: 0 });
    const stats = categoryStats.get(categoryKey);
    stats.total++;
    if (isHit) stats.hit++;

    if (!isHit) {
      mismatched.push({
        id: q.id,
        query: q.query,
        expected: [...q.expectedCubeIds],
        actual: result.cubeId,
        actualTitle: result.raw?.title ?? null,
        category: q.category
      });
    }

    if ((i + 1) % 50 === 0 || i === winnableQueries.length - 1) {
      console.log(
        `[${i + 1}/${winnableQueries.length}] running accuracy: ${formatPercent(hitCount, i + 1)}`
      );
    }

    if (delayMs > 0 && i < winnableQueries.length - 1) {
      await sleep(delayMs);
    }
  }

  // ── Build search_Serper_result.txt ──────────────────────────────────────
  const lines = [];
  lines.push("=== server.js /api/search Benchmark vs. Ground Truth ===");
  lines.push(`Target server: ${baseUrl}`);
  lines.push(`Run date: ${new Date().toISOString()}`);
  lines.push("");
  lines.push(`Distinct queries in searchResults.json: ${queries.length}`);
  lines.push(`  No ground-truth link:                 ${noLinkQueries.length}  (excluded)`);
  lines.push(`  Unwinnable (expected cube not in system): ${unwinnableQueries.length}  (excluded)`);
  lines.push(`  Winnable (used for accuracy):          ${winnableQueries.length}`);
  lines.push("");
  lines.push(`Overall accuracy: ${formatPercent(hitCount, winnableQueries.length)}`);
  lines.push("");
  lines.push("=== Accuracy by Category (expected cube's category) ===");
  lines.push("");

  const categoryRows = [...categoryStats.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  const nameWidth = Math.max(8, ...categoryRows.map(([name]) => name.length));
  lines.push(`${"Category".padEnd(nameWidth)} | Accuracy`);
  lines.push("-".repeat(nameWidth + 11));
  for (const [category, stats] of categoryRows) {
    lines.push(`${category.padEnd(nameWidth)} | ${formatPercent(stats.hit, stats.total)}`);
  }

  await fs.writeFile(TXT_OUTPUT_PATH, lines.join("\n") + "\n");
  console.log(`\nSaved ${TXT_OUTPUT_PATH}`);

  // ── Build mismatched.json ───────────────────────────────────────────────
  await fs.writeFile(MISMATCHED_OUTPUT_PATH, JSON.stringify(mismatched, null, 2));
  console.log(`Saved ${MISMATCHED_OUTPUT_PATH} (${mismatched.length} mismatches)`);

  console.log(`\nFinal accuracy: ${formatPercent(hitCount, winnableQueries.length)}`);
}

main().catch(err => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});