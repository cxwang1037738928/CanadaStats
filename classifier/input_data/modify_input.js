/**
 * process_data.js
 *
 * Pipeline:
 * 1. Load inputs.json (id, query) and classifications.json (id, classification).
 *    Both are sorted ascending by id, ids 1..N, fully aligned.
 * 2. Load query.txt and classifications.txt (one entry per line, 2295 lines each).
 * 3. Replace the first 2295 entries (by ascending id order, i.e. ids 1..2295) in
 *    inputs.json with the lines from query.txt, and the first 2295 entries in
 *    classifications.json with the lines from classifications.txt, in order.
 * 4. Combine inputs + classifications into one list of {id, query, classification}.
 * 5. Split into train/test/val (70/15/15) STRATIFIED BY CATEGORY, so each
 *    category is represented in train/test/val at roughly the same 70/15/15
 *    ratio. Order is preserved (no shuffling) within each category -- the
 *    first ~70% of each category's entries (in id order) go to train, the
 *    next ~15% to test, the remaining ~15% to val.
 * 6. Write train.json, test.json, val.json.
 *
 * Run:
 *     node process_data.js
 */

import fs from "fs";
import { readFile, writeFile } from "fs/promises";
import path from "path";

// ---------------------------------------------------------------------------
// Paths -- adjust if your files live elsewhere
// ---------------------------------------------------------------------------
const INPUTS_PATH = "./inputs.json";
const CLASSIFICATIONS_PATH = "./classifications.json";
const QUERY_TXT_PATH = "./query.txt";
const CLASS_TXT_PATH = "./classifications.txt";

const OUT_DIR = "./outputs";
const TRAIN_OUT = path.join(OUT_DIR, "train.json");
const TEST_OUT = path.join(OUT_DIR, "test.json");
const VAL_OUT = path.join(OUT_DIR, "val.json");

const N_REPLACE = 2295;
const TRAIN_FRAC = 0.70;
const TEST_FRAC = 0.15;
// remainder (~0.15) goes to val

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function readLines(filePath) {
  // Read a .txt file as one entry per line, robust to:
  // - no trailing newline on the last line
  // - Windows-style \r\n line endings
  // - stray blank lines
  const content = fs.readFileSync(filePath, "utf-8");
  return content
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split("\n")
    .filter((line) => line.trim() !== "");
}

function loadJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}

function assert(condition, message) {
  if (!condition) {
    throw new Error("Assertion failed: " + message);
  }
}

// ---------------------------------------------------------------------------
// Step 1: Load everything
// ---------------------------------------------------------------------------
console.log("Loading files...");
const inputs = loadJson(INPUTS_PATH);
const classifications = loadJson(CLASSIFICATIONS_PATH);
const newQueries = readLines(QUERY_TXT_PATH);
const newClasses = readLines(CLASS_TXT_PATH);

console.log(`  inputs.json: ${inputs.length} entries`);
console.log(`  classifications.json: ${classifications.length} entries`);
console.log(`  query.txt: ${newQueries.length} lines`);
console.log(`  classifications.txt: ${newClasses.length} lines`);

// --- Sanity checks ---
assert(
  inputs.length === classifications.length,
  `inputs.json and classifications.json have different lengths (${inputs.length} vs ${classifications.length})`
);
assert(
  inputs.every((x, i) => i === 0 || x.id >= inputs[i - 1].id),
  "inputs.json is not sorted ascending by id"
);
assert(
  classifications.every((x, i) => i === 0 || x.id >= classifications[i - 1].id),
  "classifications.json is not sorted ascending by id"
);
assert(
  inputs.every((x, i) => x.id === classifications[i].id),
  "inputs.json and classifications.json ids do not match 1:1 in order"
);
assert(
  newQueries.length === N_REPLACE,
  `Expected ${N_REPLACE} lines in query.txt, got ${newQueries.length}`
);
assert(
  newClasses.length === N_REPLACE,
  `Expected ${N_REPLACE} lines in classifications.txt, got ${newClasses.length}`
);
assert(
  inputs.length >= N_REPLACE,
  `inputs.json has fewer than ${N_REPLACE} entries (${inputs.length})`
);

// ---------------------------------------------------------------------------
// Step 2: Replace the first N_REPLACE entries (by ascending id order)
// ---------------------------------------------------------------------------
console.log(`\nReplacing first ${N_REPLACE} queries and classifications...`);
for (let i = 0; i < N_REPLACE; i++) {
  inputs[i].query = newQueries[i];
  classifications[i].classification = newClasses[i];
}

// ---------------------------------------------------------------------------
// Step 3: Combine into one list of {id, query, classification}
// ---------------------------------------------------------------------------
console.log("Combining into single dataset...");
const classById = new Map(classifications.map((c) => [c.id, c.classification]));

const combined = inputs.map((entry) => ({
  id: entry.id,
  query: entry.query,
  classification: classById.get(entry.id),
}));

console.log(`  Combined ${combined.length} entries (kept in memory, not written to disk)`);

// ---------------------------------------------------------------------------
// Step 4: Stratified split by category, 70/15/15, preserving original order
// ---------------------------------------------------------------------------
console.log("\nSplitting into train/test/val (70/15/15 stratified by category)...");

const byCategory = new Map();
for (const entry of combined) {
  if (!byCategory.has(entry.classification)) {
    byCategory.set(entry.classification, []);
  }
  byCategory.get(entry.classification).push(entry);
}

let train = [];
let test = [];
let val = [];

const splitReport = [];
const sortedCategories = [...byCategory.keys()].sort();
for (const category of sortedCategories) {
  const entries = byCategory.get(category);
  const n = entries.length;
  const nTrain = Math.floor(n * TRAIN_FRAC);
  const nTest = Math.floor(n * TEST_FRAC);
  const nVal = n - nTrain - nTest; // remainder goes to val so all entries are used

  train = train.concat(entries.slice(0, nTrain));
  test = test.concat(entries.slice(nTrain, nTrain + nTest));
  val = val.concat(entries.slice(nTrain + nTest));

  splitReport.push({ category, n, nTrain, nTest, nVal });
}

// Preserve original id order within each output split
train.sort((a, b) => a.id - b.id);
test.sort((a, b) => a.id - b.id);
val.sort((a, b) => a.id - b.id);

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.writeFileSync(TRAIN_OUT, JSON.stringify(train, null, 2), "utf-8");
fs.writeFileSync(TEST_OUT, JSON.stringify(test, null, 2), "utf-8");
fs.writeFileSync(VAL_OUT, JSON.stringify(val, null, 2), "utf-8");

console.log(`  train.json: ${train.length} entries -> ${TRAIN_OUT}`);
console.log(`  test.json:  ${test.length} entries -> ${TEST_OUT}`);
console.log(`  val.json:   ${val.length} entries -> ${VAL_OUT}`);

// ---------------------------------------------------------------------------
// Step 5: Per-category report
// ---------------------------------------------------------------------------
console.log("\nPer-category split breakdown:");
console.log(
  `  ${"category".padEnd(45)} ${"total".padStart(6)} ${"train".padStart(6)} ${"test".padStart(6)} ${"val".padStart(6)}`
);
for (const { category, n, nTrain, nTest, nVal } of splitReport) {
  console.log(
    `  ${category.padEnd(45)} ${String(n).padStart(6)} ${String(nTrain).padStart(6)} ${String(nTest).padStart(6)} ${String(nVal).padStart(6)}`
  );
}

console.log(
  `\nTotals: train=${train.length} test=${test.length} val=${val.length} ` +
    `sum=${train.length + test.length + val.length} (combined=${combined.length})`
);

assert(
  train.length + test.length + val.length === combined.length,
  "Split lost or duplicated entries!"
);
console.log("\nDone.");