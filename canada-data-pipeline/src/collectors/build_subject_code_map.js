// canada-data-pipeline/src/collectors/build_subject_code_map.js
//
// Builds subjectCodeToCategory.json — the lookup table Search_Utils.js's
// rerankByCategory() uses to map a cube's subjectCode(s) to one of the 31
// classifier categories.
//
// Join path (titles only overlap exactly, ids don't):
//   cubesMetadata.json[i].title
//     --(exact string match)-->
//   inputs.json[j].query  (only ids > 2295 are real StatCan cube titles;
//                           ids 1-2295 are synthetic search queries and are
//                           never matched against here)
//     --(inputs.json[j].id === classifications.json[k].id)-->
//   classifications.json[k].classification
//
// Every cube can carry multiple subjectCodes, and the same subjectCode is
// reused across many cubes — usually with different classifications, since
// StatCan's subject codes are coarser than this project's 31 categories.
// (On this dataset: 457 distinct subjectCodes, 304 of them — two-thirds —
// appear with more than one classification.) So each subjectCode's final
// category is decided by MAJORITY VOTE across every cube that carries it,
// not by whichever cube happened to be processed last.
//
// Output:
//   subjectCodeToCategory.json        — { [subjectCode]: category }
//     Plain string-valued map, matching what Search_Utils.js's
//     categoryForCube() already reads directly (subjectCodeMap[code]) — no
//     changes needed on the consuming side.
//
//   subjectCodeToCategory.debug.json  — { [subjectCode]: { category,
//     confidence, votes, totalVotes } }
//     Same mapping with the full vote breakdown, kept separately so the
//     plain map's format doesn't have to change. Useful for comparing
//     against StatCan's own subject hierarchy later, or for spotting weak/
//     ambiguous codes (confidence < 0.5) before trusting them.

import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const INPUTS_PATH = path.join(__dirname, "..", "..", "..", "classifier", "input_data","inputs.json");
const CLASSIFICATIONS_PATH = path.join(__dirname, "..", "..", "..", "classifier", "input_data", "classifications.json");
const CUBES_METADATA_PATH = path.join(__dirname, "cubesMetadata.json");

const OUTPUT_PATH = path.join(__dirname, "subjectCodeToCategory.json");
const DEBUG_OUTPUT_PATH = path.join(__dirname, "subjectCodeToCategory.debug.json");

// Below this majority share, a subjectCode's winning category is considered
// a weak/ambiguous signal. Not excluded from the output (per your call to
// keep everything and evaluate later) — just flagged in the debug file and
// summarized in the console output so it's visible, not silent.
const LOW_CONFIDENCE_THRESHOLD = 0.5;

async function loadJson(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  return JSON.parse(raw);
}

async function main() {
  const [inputs, classifications, cubes] = await Promise.all([
    loadJson(INPUTS_PATH),
    loadJson(CLASSIFICATIONS_PATH),
    loadJson(CUBES_METADATA_PATH)
  ]);

  // inputs.json's "id" is an internal id with no relation to cubeId — the
  // only bridge between cubesMetadata.json and classifications.json is an
  // exact title match against inputs.json's "query" field.
  const titleToId = new Map(inputs.map(item => [item.query, item.id]));
  const idToClassification = new Map(
    classifications.map(item => [item.id, item.classification])
  );

  // subjectCode -> classification -> vote count
  const votesByCode = new Map();

  let matchedCubes = 0;
  let unmatchedTitles = 0;
  let matchedButNoClassification = 0;
  let cubesWithNoSubjectCode = 0;

  for (const cube of cubes) {
    const inputId = titleToId.get(cube.title);
    if (inputId === undefined) {
      unmatchedTitles++;
      console.warn(`No inputs.json match for cube title: "${cube.title}" (cubeId ${cube.cubeId})`);
      continue;
    }

    const classification = idToClassification.get(inputId);
    if (!classification) {
      matchedButNoClassification++;
      console.warn(`No classification for inputs.json id ${inputId} (cubeId ${cube.cubeId})`);
      continue;
    }

    matchedCubes++;

    const subjectCodes = cube.subjectCode ?? [];
    if (subjectCodes.length === 0) {
      cubesWithNoSubjectCode++;
      continue;
    }

    for (const code of subjectCodes) {
      if (!votesByCode.has(code)) votesByCode.set(code, new Map());
      const codeVotes = votesByCode.get(code);
      codeVotes.set(classification, (codeVotes.get(classification) ?? 0) + 1);
    }
  }

  // Resolve each subjectCode's winning category by majority vote. Ties are
  // broken by whichever classification was encountered first for that code
  // (Map iteration order = insertion order = first-seen order across cubes)
  // — an arbitrary but deterministic tiebreak, flagged via confidence (the
  // debug file makes a tie visible: top count equals the runner-up's).
  const categoryMap = {};
  const debugMap = {};

  let lowConfidenceCount = 0;

  for (const [code, codeVotes] of votesByCode.entries()) {
    let bestCategory = null;
    let bestCount = -1;
    let totalVotes = 0;

    for (const [category, count] of codeVotes.entries()) {
      totalVotes += count;
      if (count > bestCount) {
        bestCount = count;
        bestCategory = category;
      }
    }

    const confidence = bestCount / totalVotes;
    if (confidence < LOW_CONFIDENCE_THRESHOLD) lowConfidenceCount++;

    categoryMap[code] = bestCategory;
    debugMap[code] = {
      category: bestCategory,
      confidence,
      votes: Object.fromEntries(codeVotes),
      totalVotes
    };
  }

  await fs.writeFile(OUTPUT_PATH, JSON.stringify(categoryMap, null, 2));
  await fs.writeFile(DEBUG_OUTPUT_PATH, JSON.stringify(debugMap, null, 2));

  console.log("\n=== subjectCodeToCategory build summary ===");
  console.log(`Cubes in cubesMetadata.json:        ${cubes.length}`);
  console.log(`Matched to inputs.json by title:    ${matchedCubes}`);
  console.log(`Unmatched titles (skipped):         ${unmatchedTitles}`);
  console.log(`Matched but no classification:      ${matchedButNoClassification}`);
  console.log(`Matched cubes with no subjectCode:  ${cubesWithNoSubjectCode}`);
  console.log(`Distinct subjectCodes mapped:       ${Object.keys(categoryMap).length}`);
  console.log(`  ...with majority share < ${LOW_CONFIDENCE_THRESHOLD}:     ${lowConfidenceCount}`);
  console.log(`\nSaved ${OUTPUT_PATH}`);
  console.log(`Saved ${DEBUG_OUTPUT_PATH}`);
}

main().catch(console.error);