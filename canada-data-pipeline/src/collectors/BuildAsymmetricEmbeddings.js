// BuildAsymmetricEmbeddings.js
//
// Generates cube embeddings using the asymmetric retrieval model
// (multi-qa-MiniLM-L6-cos-v1) from the already-built cubesMetadata.json.
// No StatCan API calls — reads existing metadata, runs embedding locally.
//
// Usage (run from canada-data-pipeline/src/collectors/):
//   node BuildAsymmetricEmbeddings.js [--title-dates] [--members-per-dim N]
//
//   --title-dates        embed title + dates only (no dimension names/members)
//   --members-per-dim N  max member names to include per dimension (default 8).
//                        0 = dimension names only, no member values.
//                        Ignored when --title-dates is set.
//
// Output filenames:
//   enriched, mpd=8  → cubesWithEmbeddings.multi-qa-MiniLM-L6-cos-v1.json
//   enriched, mpd=2  → cubesWithEmbeddings.multi-qa-MiniLM-L6-cos-v1.mpd2.json
//   enriched, mpd=0  → cubesWithEmbeddings.multi-qa-MiniLM-L6-cos-v1.mpd0.json
//   title-dates      → cubesWithEmbeddings.multi-qa-MiniLM-L6-cos-v1.title-dates.json

import fs from "fs/promises";
import { pipeline } from "@xenova/transformers";

const RETRIEVAL_MODEL = "Xenova/multi-qa-MiniLM-L6-cos-v1";
const METADATA_PATH   = "./cubesMetadata.json";

const GENERIC_THRESHOLD_RATIO = 0.05;

// ── Parse CLI flags ──────────────────────────────────────────────────────────

const TITLE_DATES = process.argv.includes('--title-dates');

const mpdFlagIdx = process.argv.indexOf('--members-per-dim');
const MEMBERS_PER_DIM = mpdFlagIdx !== -1 ? Number(process.argv[mpdFlagIdx + 1]) : 8;

if (!TITLE_DATES && isNaN(MEMBERS_PER_DIM)) {
  console.error('--members-per-dim requires a number');
  process.exit(1);
}

// ── Document-frequency helpers ───────────────────────────────────────────────

function buildDocumentFrequencies(cubes) {
  const memberDocFreq  = new Map();
  const dimNameDocFreq = new Map();

  for (const cube of cubes) {
    const seenMembers = new Set();
    const seenDims    = new Set();

    for (const dim of (cube.dimensions || [])) {
      if (dim.dimensionNameEn) seenDims.add(dim.dimensionNameEn);
      for (const m of (dim.memberNames || [])) seenMembers.add(m);
    }

    for (const m of seenMembers) memberDocFreq.set(m,  (memberDocFreq.get(m)  || 0) + 1);
    for (const d of seenDims)    dimNameDocFreq.set(d, (dimNameDocFreq.get(d) || 0) + 1);
  }

  return { memberDocFreq, dimNameDocFreq };
}

function buildSearchText(cube, memberDocFreq, dimNameDocFreq, totalCubes, membersPerDim) {
  const genericCutoff = Math.floor(totalCubes * GENERIC_THRESHOLD_RATIO);
  const parts = [cube.title];

  if (cube.startDate) parts.push(cube.startDate);
  if (cube.endDate)   parts.push(cube.endDate);

  for (const dim of (cube.dimensions || [])) {
    const dimName = dim.dimensionNameEn;
    if (dimName && (dimNameDocFreq.get(dimName) || 0) <= genericCutoff) {
      parts.push(dimName);
    }
    if (membersPerDim > 0) {
      const specificMembers = (dim.memberNames || [])
        .filter(m => (memberDocFreq.get(m) || 0) <= genericCutoff)
        .slice(0, membersPerDim);
      parts.push(...specificMembers);
    }
  }

  return parts.join(". ");
}

// ── Output path ──────────────────────────────────────────────────────────────

function buildOutputPath(slug) {
  if (TITLE_DATES) return `./cubesWithEmbeddings.${slug}.title-dates.json`;
  // Only encode mpd in filename when it differs from the default (8), so the
  // default run produces the expected cubesWithEmbeddings.<slug>.json filename.
  if (MEMBERS_PER_DIM !== 8) return `./cubesWithEmbeddings.${slug}.mpd${MEMBERS_PER_DIM}.json`;
  return `./cubesWithEmbeddings.${slug}.json`;
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const slug    = RETRIEVAL_MODEL.split("/").pop();
  const outPath = buildOutputPath(slug);

  console.log(`Asymmetric Embeddings Builder`);
  console.log(`  Model:           ${RETRIEVAL_MODEL}`);
  console.log(`  Text mode:       ${TITLE_DATES ? 'title-dates' : 'enriched'}`);
  if (!TITLE_DATES) console.log(`  Members per dim: ${MEMBERS_PER_DIM}`);
  console.log(`  Output:          ${outPath}`);
  console.log("=".repeat(60));

  const cubes = JSON.parse(await fs.readFile(METADATA_PATH, "utf8"));
  console.log(`Loaded ${cubes.length} cubes from ${METADATA_PATH}`);

  let getDocText;
  if (TITLE_DATES) {
    getDocText = cube => `${cube.title} ${cube.startDate || ''} ${cube.endDate || ''}`;
    console.log(`\nSample title-dates text:\n  "${getDocText(cubes[0])}"\n`);
  } else {
    const { memberDocFreq, dimNameDocFreq } = buildDocumentFrequencies(cubes);
    const genericCutoff = Math.floor(cubes.length * GENERIC_THRESHOLD_RATIO);
    const genericMemberCount = [...memberDocFreq.values()].filter(n => n > genericCutoff).length;
    console.log(`  ${memberDocFreq.size} member names — ${genericMemberCount} generic filtered, ${memberDocFreq.size - genericMemberCount} kept`);
    const sample = buildSearchText(cubes[0], memberDocFreq, dimNameDocFreq, cubes.length, MEMBERS_PER_DIM);
    console.log(`\nSample enriched text (mpd=${MEMBERS_PER_DIM}):\n  "${sample.slice(0, 300)}..."\n`);
    getDocText = cube => buildSearchText(cube, memberDocFreq, dimNameDocFreq, cubes.length, MEMBERS_PER_DIM);
  }

  console.log(`Loading ${RETRIEVAL_MODEL}…`);
  const extractor = await pipeline("feature-extraction", RETRIEVAL_MODEL);
  console.log(`Generating embeddings for ${cubes.length} cubes…`);

  const results = [];

  for (let i = 0; i < cubes.length; i++) {
    const cube = cubes[i];
    const text = getDocText(cube);
    const emb  = await extractor(text, { pooling: "mean", normalize: true });

    results.push({
      cubeId:         cube.cubeId,
      title:          cube.title,
      startDate:      cube.startDate,
      endDate:        cube.endDate,
      frequencyCode:  cube.frequencyCode,
      geographyCount: cube.geographyCount,
      embedding:      Array.from(emb.data).map(v => Math.round(v * 1000) / 1000),
    });

    if ((i + 1) % 100 === 0 || i + 1 === cubes.length) {
      console.log(`  Embedded ${i + 1}/${cubes.length}`);
    }
  }

  await fs.writeFile(outPath, JSON.stringify(results, null, 2));
  const stats = await fs.stat(outPath);
  const dims  = results[0]?.embedding.length ?? 0;
  console.log(`\n✓ Saved ${outPath}`);
  console.log(`  ${results.length} cubes · ${dims} dimensions · ${(stats.size / (1024 * 1024)).toFixed(2)} MB`);
}

main().catch(console.error);
