// backend/Search_Utils.js
//
// Helper module for the /api/search endpoint in server.js. Wraps three things
// that used to live (or were missing) inline in server.js:
//
//   1. Loading the trained MLP query classifier (saved by
//      classifier/train_mlp.js) without depending on @tensorflow/tfjs-node.
//   2. Re-ranking cosine-similarity search results with a category-match
//      boost, using the MLP's prediction for the user's query plus a
//      subjectCode -> category lookup table for each cube.
//   3. Enriching search results with the richer fields available in
//      cubesMetadata.json (footnotes, full dimension names/members,
//      subjectCode, archive status, etc.) that aren't present in the
//      slimmer cubesWithEmbeddings.*.json files used for the similarity
//      search itself.
//
// server.js stays responsible for: loading the embedding model, computing
// the query embedding, the cosine similarity ranking pass, and the StatCan
// metadata fetch for the winning cube. This module sits between the
// similarity ranking and the final response.

import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import * as tf from "@tensorflow/tfjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ── Paths ──────────────────────────────────────────────────────────────────
// All paths are resolved relative to this file's location so Search_Utils.js
// works regardless of which directory the Node process is launched from
// (mirrors the dual-path resolution pattern already used in server.js).

const PROJECT_ROOT = path.join(__dirname, "..");

const MLP_DIR = path.join(PROJECT_ROOT, "classifier", "query-classifier");
const METADATA_PATH = path.join(
  PROJECT_ROOT,
  "canada-data-pipeline",
  "src",
  "collectors",
  "cubesMetadata.json"
);
// The subjectCode -> category lookup table is built separately (per your
// classification pipeline) and is expected to live alongside the other
// collector outputs. Shape: { [subjectCode: string]: categoryName }.
const SUBJECT_CODE_MAP_PATH = path.join(
  PROJECT_ROOT,
  "canada-data-pipeline",
  "src",
  "collectors",
  "subjectCodeToCategory.json"
);
// Top category-distinctive keywords per category, built by
// canada-data-pipeline/src/collectors/build_category_keywords.js from
// title-only TF-IDF scoring. Shape: { [categoryName]: string[] }.
const CATEGORY_KEYWORDS_PATH = path.join(
  PROJECT_ROOT,
  "canada-data-pipeline",
  "src",
  "collectors",
  "categoryKeywords.json"
);

// The embedding model used for the search endpoint and the MLP classifier
// must match — the MLP was trained on vectors from this specific model.
export const CLASSIFIER_EMBEDDING_MODEL = "Xenova/all-MiniLM-L12-v2";

const CATEGORIES = [
  "Government",
  "Income, pensions, spending and wealth",
  "International trade",
  "Health",
  "Labour",
  "Languages",
  "Manufacturing",
  "Population and demography",
  "Prices and price indexes",
  "Statistical methods",
  "Retail and wholesale",
  "Business and consumer services and culture",
  "Digital economy and society",
  "Transportation",
  "Travel and tourism",
  "Energy",
  "Science and technology",
  "Agriculture and food",
  "Business performance and ownership",
  "Construction",
  "Crime and justice",
  "Economic accounts",
  "Education, training and learning",
  "Environment",
  "Families, households and marital status",
  "Indigenous peoples",
  "Children and youth",
  "Immigration and ethnocultural diversity",
  "Older adults and population aging",
  "Society and community",
  "Housing"
];

// How much to multiply cosine similarity by when a cube's mapped category
// matches the MLP's predicted category for the query. Re-ranking only
// (never excludes a cube outright) per the chosen design — cubes that don't
// match are left at their original similarity score.
const CATEGORY_BOOST_FACTOR = 1.15;
const ARCHIVE_PENALTY = 0.85;

// How much to multiply similarity by, per matched keyword, when the user's
// query contains one of a cube's category's top distinctive keywords (see
// build_category_keywords.js). Multiplicative and stacking, same mechanism
// as the category boost — e.g. two keyword matches compound to ~1.05^2.
// Capped via MAX_KEYWORD_MATCHES so a long query packed with a category's
// entire keyword list can't compound into an unbounded multiplier.
const KEYWORD_BOOST_FACTOR = 1.05;
const MAX_KEYWORD_MATCHES = 5;

function safeModelKey(modelId) {
  return modelId.replace(/[\/\\]/g, "_");
}

// ── In-memory caches (populated lazily, mirrors server.js's cachedCubes
// pattern so repeated requests don't re-read/re-parse files or reload the
// model) ─────────────────────────────────────────────────────────────────

let cachedMetadataById = null;
let cachedSubjectCodeMap = null;
let cachedCategoryKeywords = null;
let cachedMlpModel = null;

// ── MLP loading ────────────────────────────────────────────────────────────
//
// Mirrors classifier/train_mlp.js's save format exactly: two JSON files,
// mlp-model__<key>.json (raw model topology) and mlp-weights__<key>.json
// ({ weightSpecs, weightDataBase64 }). Reconstructed via tf.io.fromMemory
// rather than tf.loadLayersModel('file://...'), since this project doesn't
// depend on @tensorflow/tfjs-node (which is required for the file://
// IOHandler) — see classifier/train_mlp.js for the matching save logic and
// prior discussion of why model.toJSON() alone does not produce loadable
// topology.
async function loadMLP(modelId) {
  const key = safeModelKey(modelId);

  const modelTopology = JSON.parse(
    await fs.readFile(path.join(MLP_DIR, `mlp-model__${key}.json`), "utf8")
  );
  const { weightSpecs, weightDataBase64 } = JSON.parse(
    await fs.readFile(path.join(MLP_DIR, `mlp-weights__${key}.json`), "utf8")
  );
  const weightData = Uint8Array.from(Buffer.from(weightDataBase64, "base64")).buffer;

  return tf.loadLayersModel(
    tf.io.fromMemory({ modelTopology, weightSpecs, weightData })
  );
}

// Loads (or returns the cached) MLP classifier. Call this once at server
// startup, same as getEmbeddingModel() in server.js, so the first search
// request isn't slowed down by a cold load.
export async function getMlpModel() {
  if (!cachedMlpModel) {
    console.log(`Loading MLP classifier for ${CLASSIFIER_EMBEDDING_MODEL}…`);
    cachedMlpModel = await loadMLP(CLASSIFIER_EMBEDDING_MODEL);
    console.log("MLP classifier ready");
  }
  return cachedMlpModel;
}

// Predicts a category for a single already-computed query embedding vector
// (plain number[], same shape as what server.js gets from
// Array.from(emb.data)). Returns { category, confidence }.
//
// Takes the embedding vector rather than the raw query string so server.js
// can reuse the same embedding it already computed for the cosine similarity
// search — no need to embed the query twice, as long as the embedding model
// used for that search matches CLASSIFIER_EMBEDDING_MODEL.
export async function classifyQueryEmbedding(queryEmbedding) {
  const model = await getMlpModel();
  const inputTensor = tf.tensor2d([queryEmbedding], [1, queryEmbedding.length]);

  try {
    const prediction = model.predict(inputTensor);
    const probs = await prediction.data();
    prediction.dispose();

    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }

    return {
      category: CATEGORIES[bestIdx] ?? null,
      confidence: probs[bestIdx]
    };
  } finally {
    inputTensor.dispose();
  }
}

// ── Metadata loading ───────────────────────────────────────────────────────

async function loadMetadataById() {
  if (cachedMetadataById) return cachedMetadataById;

  const raw = await fs.readFile(METADATA_PATH, "utf8");
  const cubes = JSON.parse(raw);

  cachedMetadataById = new Map(cubes.map(c => [String(c.cubeId), c]));
  console.log(`Loaded metadata for ${cachedMetadataById.size} cubes.`);
  return cachedMetadataById;
}

async function loadCategoryKeywords() {
  if (cachedCategoryKeywords) return cachedCategoryKeywords;

  try {
    const raw = await fs.readFile(CATEGORY_KEYWORDS_PATH, "utf8");
    cachedCategoryKeywords = JSON.parse(raw);
    console.log(
      `Loaded category keywords (${Object.keys(cachedCategoryKeywords).length} categories).`
    );
  } catch (err) {
    console.warn(
      `categoryKeywords.json not found or unreadable at ${CATEGORY_KEYWORDS_PATH} — ` +
      `keyword boosting will be skipped until it exists. (${err.message})`
    );
    cachedCategoryKeywords = {};
  }
  return cachedCategoryKeywords;
}

// Same tokenization rule used to build categoryKeywords.json (lowercase,
// letters-only words) so query words can actually match the extracted
// keywords. Intentionally does NOT apply the stopword/min-length filtering
// from build_category_keywords.js — those only matter for keyword
// *extraction* (keeping noise out of the keyword lists themselves); a short
// or common word typed by the user should still count as a match if it
// happens to appear in a category's keyword list.
function tokenizeQuery(query) {
  return new Set((query.toLowerCase().match(/[a-z]+/g)) ?? []);
}

// Counts how many of a category's top keywords appear in the (already
// tokenized) query, capped at MAX_KEYWORD_MATCHES.
function countKeywordMatches(queryWords, category, categoryKeywords) {
  const keywords = categoryKeywords[category];
  if (!keywords?.length) return 0;

  let matches = 0;
  for (const keyword of keywords) {
    if (queryWords.has(keyword)) {
      matches++;
      if (matches >= MAX_KEYWORD_MATCHES) break;
    }
  }
  return matches;
}


async function loadSubjectCodeMap() {
  if (cachedSubjectCodeMap) return cachedSubjectCodeMap;

  try {
    const raw = await fs.readFile(SUBJECT_CODE_MAP_PATH, "utf8");
    cachedSubjectCodeMap = JSON.parse(raw);
    console.log(
      `Loaded subjectCode→category map (${Object.keys(cachedSubjectCodeMap).length} codes).`
    );
  } catch (err) {
    console.warn(
      `subjectCodeToCategory.json not found or unreadable at ${SUBJECT_CODE_MAP_PATH} — ` +
      `category boosting will be skipped until it exists. (${err.message})`
    );
    cachedSubjectCodeMap = {};
  }
  return cachedSubjectCodeMap;
}

// Returns the first known category among a cube's subjectCode list, or null
// if none of its codes are in the map (e.g. map not built yet, or codes not
// covered).
function categoryForCube(cube, subjectCodeMap) {
  if (!cube?.subjectCode?.length) return null;
  for (const code of cube.subjectCode) {
    const category = subjectCodeMap[code];
    if (category) return category;
  }
  return null;
}

// ── Re-ranking ─────────────────────────────────────────────────────────────
//
// Takes the cosine-similarity-ranked candidates server.js already produced
// (each { cubeId, title, similarity, ... }), and applies two independent,
// stacking multiplicative boosts:
//
//   1. Category match: boosts cubes whose mapped category (via
//      subjectCodeToCategory.json) matches the MLP's predicted category for
//      the query embedding.
//   2. Keyword match: boosts cubes whose mapped category's top distinctive
//      keywords (via categoryKeywords.json) appear in the raw query text —
//      e.g. typing "police offences" boosts Crime and justice cubes even if
//      the MLP's top-1 prediction landed on a different category, since
//      keyword matching is independent of the MLP's category prediction.
//
// `query` is the original query string (needed for keyword matching, which
// works on raw text — the embedding alone can't be tokenized back into
// words). `queryEmbedding` is reused for the MLP category prediction so the
// caller doesn't need to embed the query twice.
//
// Cubes with no resolvable category (map miss, or no subjectCode) are left
// at their original score for both boosts — boosting is opportunity, not a
// filter, so nothing is excluded if a mapping is incomplete.
export async function rerankByCategory(rankedCandidates, query, queryEmbedding) {
  const [subjectCodeMap, categoryKeywords, metadataById, predicted] = await Promise.all([
    loadSubjectCodeMap(),
    loadCategoryKeywords(),
    loadMetadataById(),
    classifyQueryEmbedding(queryEmbedding)
  ]);

  const queryWords = tokenizeQuery(query);

  const boosted = rankedCandidates.map(candidate => {
    const cube = metadataById.get(String(candidate.cubeId));
    const cubeCategory = cube ? categoryForCube(cube, subjectCodeMap) : null;

    const matchesPredictedCategory =
      predicted.category !== null && cubeCategory === predicted.category;
    const keywordMatches = cubeCategory
      ? countKeywordMatches(queryWords, cubeCategory, categoryKeywords)
      : 0;
    
    // archived cubes (archiveStatusCode !== "2") are penalized across the board, since they're less likely to be relevant to a current query — but the penalty is milder for cubes that match the predicted category, since relevance is relevance even if archived. The penalty is applied before category/keyword boosts so it doesn't get multiplied by them — e.g. an archived cube that matches the predicted category still gets the full category boost, just off a lower base similarity.
    const isCurrent = cube?.archiveStatusCode === "2";
    const archivePenalty = isCurrent ? 1 : ARCHIVE_PENALTY; // tuned by testing different numbers
    const categoryMultiplier = matchesPredictedCategory ? CATEGORY_BOOST_FACTOR : 1;
    const keywordMultiplier = Math.pow(KEYWORD_BOOST_FACTOR, keywordMatches);

    return {
      ...candidate,
      cubeCategory,
      keywordMatches,
      boostedSimilarity: candidate.similarity * categoryMultiplier * keywordMultiplier * archivePenalty
    };
  });

  boosted.sort((a, b) => b.boostedSimilarity - a.boostedSimilarity);

  return {
    reranked: boosted,
    predictedCategory: predicted.category,
    predictedCategoryConfidence: predicted.confidence
  };
}

// ── Response enrichment ─────────────────────────────────────────────────────
//
// Merges in the extra fields from cubesMetadata.json that aren't present in
// the slim cubesWithEmbeddings.*.json used for the similarity search —
// footnotes, full dimension names/members, subjectCode, survey codes,
// archive status, series/datapoint counts, and bilingual titles. Returns
// null fields rather than omitting keys if metadata for a cube can't be
// found, so the response shape stays predictable for the frontend.
export async function enrichWithMetadata(cubeId) {
  const metadataById = await loadMetadataById();
  const cube = metadataById.get(String(cubeId));

  if (!cube) {
    console.warn(`No metadata entry found for cube ${cubeId} during enrichment.`);
    return {
      titleFr: null,
      releaseTime: null,
      archiveStatusCode: null,
      archiveStatusEn: null,
      nbSeriesCube: null,
      nbDatapointsCube: null,
      surveyCode: [],
      subjectCode: [],
      footnotes: [],
      dimensions: []
    };
  }

  return {
    titleFr: cube.titleFr ?? null,
    releaseTime: cube.releaseTime ?? null,
    archiveStatusCode: cube.archiveStatusCode ?? null,
    archiveStatusEn: cube.archiveStatusEn ?? null,
    nbSeriesCube: cube.nbSeriesCube ?? null,
    nbDatapointsCube: cube.nbDatapointsCube ?? null,
    surveyCode: cube.surveyCode ?? [],
    subjectCode: cube.subjectCode ?? [],
    footnotes: cube.footnotes ?? [],
    dimensions: cube.dimensions ?? []
  };
}

// Convenience: whether a cube counts as "current" (not archived), useful if
// server.js wants to filter or flag archived tables. Exposed separately
// rather than baked into enrichWithMetadata so callers choose whether to use
// it — archived cubes may still be relevant results, just worth flagging.
export function isCurrentCube(metadataFields) {
  return metadataFields?.archiveStatusCode === "2";
}