// canada-data-pipeline/src/collectors/build_category_keywords.js
//
// Builds categoryKeywords.json — the top 10-20 most category-distinctive
// words for each of the 31 classifier categories, used by Search_Utils.js
// to apply a small per-keyword multiplicative boost when a user's query
// contains words strongly associated with a cube's category.
//
// Join path (same as build_subject_code_map.js):
//   cubesMetadata.json[i].title
//     --(exact string match)-->
//   inputs.json[j].query  (only ids > 2295 are real StatCan cube titles)
//     --(inputs.json[j].id === classifications.json[k].id)-->
//   classifications.json[k].classification
//
// SCORING: title-only, TF-IDF-weighted — NOT raw frequency, and NOT
// title+dimensions. Both alternatives were tested against this dataset and
// rejected:
//
//   - Raw frequency, title+dimensions: dominated entirely by generic
//     statistical-table vocabulary that appears across every category
//     (e.g. top "Health" words were "number", "confidence", "interval",
//     "age", "sex" — StatCan's standard demographic breakdown columns, not
//     actual health vocabulary).
//   - Dimension member names specifically are mostly standardized breakdown
//     categories (age groups, sex, confidence intervals, "both sexes", high/
//     low estimates) reused across nearly every table regardless of subject
//     — they add noise, not signal, for this purpose.
//   - Title-only + TF-IDF (word frequency within a category, downweighted
//     by how many *other* categories also use that word) reliably surfaces
//     genuinely distinctive vocabulary: "police/offences/crimes" for Crime
//     and justice, "tuition/student/school" for Education, "vehicle/
//     trucking/kilometres" for Transportation, etc.
//
// TF-IDF here is computed at the CATEGORY level (not per-document): for
// each category, "term frequency" = (# cubes in that category whose title
// contains the word) / (# cubes in that category), and "inverse document
// frequency" = log(31 categories / (1 + # categories where the word
// appears at all)). A word that's common within a category but also common
// everywhere else scores low; a word concentrated in one or two categories
// scores high.

import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const INPUTS_PATH = path.join(__dirname, "..", "..", "..", "classifier", "input_data","inputs.json");
const CLASSIFICATIONS_PATH = path.join(__dirname, "..", "..", "..", "classifier", "input_data", "classifications.json");
const CUBES_METADATA_PATH = path.join(__dirname, "cubesMetadata.json");
const OUTPUT_PATH = path.join(__dirname, "categoryKeywords.json");

// How many top keywords to keep per category.
const TOP_N = 15;

// Words to ignore entirely — common English stopwords plus StatCan-specific
// boilerplate that shows up across titles regardless of subject matter
// (unit-of-measure suffixes, footnote-style phrasing like "unless otherwise
// noted", classification-scheme leftovers like "not elsewhere classified").
const STOPWORDS = new Set([
  "a", "an", "the", "of", "for", "and", "to", "in", "on", "by", "with",
  "from", "at", "as", "is", "are", "was", "were", "their", "its", "this",
  "that", "other", "total", "all", "not", "elsewhere", "classified",
  "n", "e", "c", "unless", "otherwise", "noted", "number", "dollars",
  "persons", "percentage"
]);

const MIN_WORD_LENGTH = 3; // gets rid of small words like "of", "to", "by" that aren't in the stopword list but are still generic and unhelpful.

async function loadJson(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  return JSON.parse(raw);
}

function tokenize(text) {
  const words = (text.toLowerCase().match(/[a-z]+/g)) ?? [];
  return words.filter(w => w.length > MIN_WORD_LENGTH - 1 && !STOPWORDS.has(w));
}

async function main() {
  const [inputs, classifications, cubes] = await Promise.all([
    loadJson(INPUTS_PATH),
    loadJson(CLASSIFICATIONS_PATH),
    loadJson(CUBES_METADATA_PATH)
  ]);

  const titleToId = new Map(inputs.map(item => [item.query, item.id]));
  const idToClassification = new Map(
    classifications.map(item => [item.id, item.classification])
  );

  // category -> word -> number of cubes in that category whose title
  // contains the word (set-based per cube, so a repeated word within one
  // title only counts once — this matches the "is this word characteristic
  // of titles in this category" question, not "how often does it appear").
  const categoryWordCubeCounts = new Map();
  // category -> total number of cubes counted for that category
  const categoryCubeTotals = new Map();

  let matchedCubes = 0;
  let unmatchedTitles = 0;

  for (const cube of cubes) {
    const inputId = titleToId.get(cube.title);
    if (inputId === undefined) {
      unmatchedTitles++;
      continue;
    }

    const classification = idToClassification.get(inputId);
    if (!classification) continue;

    matchedCubes++;

    if (!categoryWordCubeCounts.has(classification)) {
      categoryWordCubeCounts.set(classification, new Map());
    }
    categoryCubeTotals.set(
      classification,
      (categoryCubeTotals.get(classification) ?? 0) + 1
    );

    const wordCounts = categoryWordCubeCounts.get(classification);
    const uniqueWords = new Set(tokenize(cube.title));
    for (const word of uniqueWords) {
      wordCounts.set(word, (wordCounts.get(word) ?? 0) + 1);
    }
  }

  const categories = [...categoryWordCubeCounts.keys()];
  const numCategories = categories.length;

  // Category-level document frequency: how many categories does each word
  // appear in at all (regardless of how often within each).
  const wordCategoryDocFreq = new Map();
  for (const wordCounts of categoryWordCubeCounts.values()) {
    for (const word of wordCounts.keys()) {
      wordCategoryDocFreq.set(word, (wordCategoryDocFreq.get(word) ?? 0) + 1);
    }
  }

  const categoryKeywords = {};

  for (const category of categories) {
    const wordCounts = categoryWordCubeCounts.get(category);
    const cubeTotal = categoryCubeTotals.get(category);

    const scored = [...wordCounts.entries()].map(([word, count]) => {
      const tf = count / cubeTotal;
      const idf = Math.log(numCategories / (1 + wordCategoryDocFreq.get(word)));
      return { word, score: tf * idf, count };
    });

    scored.sort((a, b) => b.score - a.score);

    categoryKeywords[category] = scored.slice(0, TOP_N).map(s => s.word);
  }

  await fs.writeFile(OUTPUT_PATH, JSON.stringify(categoryKeywords, null, 2));

  console.log("\n=== categoryKeywords build summary ===");
  console.log(`Cubes in cubesMetadata.json:     ${cubes.length}`);
  console.log(`Matched to inputs.json by title: ${matchedCubes}`);
  console.log(`Unmatched titles (skipped):      ${unmatchedTitles}`);
  console.log(`Categories with keywords:        ${categories.length}`);
  console.log(`\nSample:`);
  for (const category of categories.slice(0, 3)) {
    console.log(`  ${category}: ${categoryKeywords[category].slice(0, 8).join(", ")}`);
  }
  console.log(`\nSaved ${OUTPUT_PATH}`);
}

main().catch(console.error);