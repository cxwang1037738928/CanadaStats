/**
 * classifier/generate_input.js
 *
 * Fetches all StatCan cube titles via getAllCubesListLite, then writes
 * inputs.json as:
 *   [ ...query.json entries (ids 1–2295), ...cube entries (ids 2296+) ]
 *
 * Each object has the shape: { id: number, query: string }
 *
 * Rate-limiter: starts at MIN_DELAY_MS between requests and scales up
 * linearly toward MAX_DELAY_MS as the total request count grows, so the
 * script stays polite to the StatCan API regardless of how many cubes
 * are returned.
 */

import fs from "fs";
import 'dotenv/config';

// ─── Files ────────────────────────────────────────────────────────────────────
const QUERY_FILE  = "query.json";   // existing 2295 entries  [{ id, query }]
const OUTPUT_FILE = "inputs.json";  // merged output

// ─── StatCan API ──────────────────────────────────────────────────────────────
const STATCAN_URL = "https://www150.statcan.gc.ca/t1/wds/rest/getAllCubesListLite";

// ─── ID offset ────────────────────────────────────────────────────────────────
// query.json occupies ids 1–2295; cubes start at 2296
const CUBE_ID_START = 2296;

// ─── Rate limiter config ──────────────────────────────────────────────────────
const MIN_DELAY_MS   = 80;    // minimum pause between requests (ms)
const MAX_DELAY_MS   = 800;   // ceiling pause once request count is high
const SCALE_EVERY    = 50;    // add one ramp step every N requests
const RAMP_MS        = 20;    // ms added per ramp step

/**
 * Returns the delay to wait after `requestCount` total requests have been made.
 * Scales linearly from MIN_DELAY_MS up to MAX_DELAY_MS.
 */
function delayFor(requestCount) {
  const steps = Math.floor(requestCount / SCALE_EVERY);
  return Math.min(MIN_DELAY_MS + steps * RAMP_MS, MAX_DELAY_MS);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─── Fetch helpers ────────────────────────────────────────────────────────────

/**
 * Fetches the full cube list in one request (getAllCubesListLite is a single
 * GET that returns the whole catalogue — no pagination needed).
 * The rate-limiter is applied *after* this single call so subsequent
 * per-cube calls (if added later) stay polite.
 */
async function fetchCubeList(requestCounter) {
  const delay = delayFor(requestCounter.count);
  requestCounter.count++;

  const res = await fetch(STATCAN_URL, {
    headers: { Accept: "application/json" },
  });

  if (!res.ok) {
    throw new Error(`StatCan API error: ${res.status} ${res.statusText}`);
  }

  const json = await res.json();

  // Honour the rate-limit delay after the response arrives
  await sleep(delay);

  return json;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  // 1. Load query.json
  if (!fs.existsSync(QUERY_FILE)) {
    console.error(`Error: '${QUERY_FILE}' not found.`);
    process.exit(1);
  }

  const queryEntries = JSON.parse(fs.readFileSync(QUERY_FILE, "utf-8"));

  if (!Array.isArray(queryEntries) || queryEntries.length === 0) {
    console.error("Error: query.json is empty or not an array.");
    process.exit(1);
  }

  console.log(`Loaded ${queryEntries.length} existing entries from '${QUERY_FILE}'.`);

  // Normalise to { id, query } shape (query.json uses the same fields, but guard anyway)
  const baseEntries = queryEntries.map(({ id, query }) => ({ id, query }));

  // 2. Fetch StatCan cube list
  console.log(`\nFetching cube list from StatCan API…`);
  console.log(`  URL: ${STATCAN_URL}`);

  const requestCounter = { count: 0 };
  let rawCubes;

  try {
    rawCubes = await fetchCubeList(requestCounter);
  } catch (err) {
    console.error(`Failed to fetch cube list: ${err.message}`);
    process.exit(1);
  }

  // getAllCubesListLite returns either a plain array or { status, object: [...] }
  const cubeArray = Array.isArray(rawCubes)
    ? rawCubes
    : Array.isArray(rawCubes?.object)
      ? rawCubes.object
      : null;

  if (!cubeArray) {
    console.error("Unexpected response shape from StatCan API:", JSON.stringify(rawCubes).slice(0, 300));
    process.exit(1);
  }

  console.log(`  Received ${cubeArray.length} cube records.`);

  // 3. Map cubes → { id, query }
  //    Each lite cube record has: productId, cubeTitleEn, cubeTitleFr, …
  let skipped = 0;
  const cubeEntries = [];

  cubeArray.forEach((cube, index) => {
    const title = cube.cubeTitleEn?.trim();
    if (!title) {
      skipped++;
      return;
    }
    cubeEntries.push({
      id: CUBE_ID_START + index,
      query: title,
    });
  });

  if (skipped) {
    console.log(`  Skipped ${skipped} cube(s) with missing English title.`);
  }

  // 4. Merge and write
  const merged = [...baseEntries, ...cubeEntries];

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(merged, null, 2), "utf-8");

  console.log(`\nDone.`);
  console.log(`  query.json entries : ${baseEntries.length}  (ids ${baseEntries[0]?.id}–${baseEntries.at(-1)?.id})`);
  console.log(`  StatCan cube entries: ${cubeEntries.length}  (ids ${cubeEntries[0]?.id}–${cubeEntries.at(-1)?.id})`);
  console.log(`  Total written to '${OUTPUT_FILE}': ${merged.length} entries`);
}

main().catch((err) => {
  console.error("Unexpected error:", err);
  process.exit(1);
});