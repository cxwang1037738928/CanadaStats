// buildProvincialCubesWithEmbeddings.js
//
// 1. Fetches all StatCan cubes
// 2. Filters to only cubes that include all 10 Canadian provinces in the geography
// 3. Extracts richer metadata for each matching cube (footnotes, non-geography
//    dimensions/members, dates, frequency, etc.) and saves it to cubesMetadata.json
// 4. Generates embeddings for "title + startDate + endDate" using 5 different
//    Xenova embedding models, saving one output file per model
//

import axios from "axios";
import fs from "fs/promises";
import { pipeline } from "@xenova/transformers";

const BASE_URL = "https://www150.statcan.gc.ca/t1/wds/rest";

const REQUIRED_PROVINCES = [
  "Newfoundland and Labrador",
  "Prince Edward Island",
  "Nova Scotia",
  "New Brunswick",
  "Quebec",
  "Ontario",
  "Manitoba",
  "Saskatchewan",
  "Alberta",
  "British Columbia",
];

// Embedding models to generate separate embedding files for.
const EMBEDDING_MODELS = [
  "Xenova/all-MiniLM-L6-v2",
  "Xenova/all-MiniLM-L12-v2",
  "Xenova/paraphrase-MiniLM-L3-v2",
  "Xenova/all-mpnet-base-v2",
  "Xenova/bge-small-en-v1.5",
];

const METADATA_OUT_PATH = "./cubesMetadata.json";

// -----------------------------
// Fetch all cubes
// -----------------------------
async function getAllCubes() {
  const url = `${BASE_URL}/getAllCubesListLite`;
  const response = await axios.get(url, {
    headers: { Accept: "application/json" },
  });
  console.log(`Fetched ${response.data?.length || 0} cubes total`);
  return response.data || [];
}

// -----------------------------
// Fetch cube metadata
// -----------------------------
async function getCubeMetadata(productId) {
  const url = `${BASE_URL}/getCubeMetadata`;

  try {
    const response = await axios({
      method: "post",
      url,
      data: [{ productId }],
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      timeout: 15000,
    });

    if (Array.isArray(response.data) && response.data.length > 0) {
      const result = response.data[0];
      if (result?.object) return result.object;
    }

    return null;
  } catch (err) {
    if (err.response?.status === 404 || err.response?.status === 406) return null;
    console.error(`Failed metadata for ${productId}: ${err.message}`);
    return null;
  }
}

// -----------------------------
// Find the Geography dimension object (raw, from metadata.dimension)
// -----------------------------
function findGeographyDimension(metadata) {
  if (!metadata) return null;
  const dimensions = metadata.dimension;
  if (!Array.isArray(dimensions)) return null;

  return (
    dimensions.find((dim) => {
      const name = dim.dimensionNameEn || "";
      return name === "Geography" || name.includes("Geography");
    }) || null
  );
}

// -----------------------------
// Extract geography member names from metadata
// -----------------------------
function getGeographyMembers(metadata) {
  const geographyDim = findGeographyDimension(metadata);
  if (!geographyDim) return [];

  const members = geographyDim.member;
  if (!Array.isArray(members)) return [];

  return members
    .map((m) => m.memberNameEn || m.memberName)
    .filter(Boolean);
}

// -----------------------------
// Check if all 10 provinces are present
// -----------------------------
function containsAllProvinces(geographies) {
  if (!geographies.length) return false;
  const lower = geographies.map((g) => g.toLowerCase());
  return REQUIRED_PROVINCES.every((p) => lower.includes(p.toLowerCase()));
}

// -----------------------------
// Extract non-geography dimensions: just name + member names (skip the
// heavy per-member classification/vintage/etc. fields to keep this light)
// -----------------------------
function getNonGeographyDimensions(metadata) {
  if (!metadata) return [];
  const dimensions = metadata.dimension;
  if (!Array.isArray(dimensions)) return [];

  return dimensions
    .filter((dim) => {
      const name = dim.dimensionNameEn || "";
      return !(name === "Geography" || name.includes("Geography"));
    })
    .map((dim) => {
      const members = Array.isArray(dim.member) ? dim.member : [];
      return {
        dimensionNameEn: dim.dimensionNameEn || null,
        dimensionNameFr: dim.dimensionNameFr || null,
        memberNames: members
          .map((m) => m.memberNameEn || m.memberName)
          .filter(Boolean),
      };
    });
}

// -----------------------------
// Flatten footnotes into a plain list of English footnote strings
// -----------------------------
function getFootnotes(metadata) {
  if (!metadata) return [];
  const footnotes = metadata.footnote;
  if (!Array.isArray(footnotes)) return [];

  return footnotes
    .map((f) => f.footnotesEn)
    .filter(Boolean);
}

// -----------------------------
// Verify API connectivity with a known cube
// -----------------------------
async function testApi() {
  console.log("Testing API with cube 35100003...");
  const metadata = await getCubeMetadata(35100003);
  if (metadata) {
    console.log(` API OK — "${metadata.cubeTitleEn}"`);
    return true;
  }
  console.log("  API test failed");
  return false;
}

// -----------------------------
// Find all provincial cubes and build their full metadata records
// -----------------------------
async function findProvincialCubes(cubes, maxToCheck = 9000) {
  const limit = Math.min(cubes.length, maxToCheck);
  const provincialCubes = [];

  for (let i = 0; i < limit; i++) {
    const cube = cubes[i];

    const metadata = await getCubeMetadata(cube.productId);

    if (metadata) {
      const geographies = getGeographyMembers(metadata);

      if (containsAllProvinces(geographies)) {
        console.log(`\n✓ ${cube.productId}: ${cube.cubeTitleEn}`);

        provincialCubes.push({
          cubeId: String(cube.productId),
          title: metadata.cubeTitleEn,
          titleFr: metadata.cubeTitleFr,
          startDate: metadata.cubeStartDate,
          endDate: metadata.cubeEndDate,
          releaseTime: metadata.releaseTime,
          frequencyCode: metadata.frequencyCode,
          archiveStatusCode: metadata.archiveStatusCode,
          archiveStatusEn: metadata.archiveStatusEn,
          nbSeriesCube: metadata.nbSeriesCube,
          nbDatapointsCube: metadata.nbDatapointsCube,
          geographyCount: geographies.length,
          surveyCode: metadata.surveyCode || [],
          subjectCode: metadata.subjectCode || [],
          footnotes: getFootnotes(metadata),
          dimensions: getNonGeographyDimensions(metadata),
        });
      }
    }
    const processed = i + 1;
    // print a console log everytime 30 cubes are processed, or at the end of the loop
    if (processed % 30 === 0 || processed === limit) {
      console.log(
        `Progress: ${processed}/${limit} checked — ${provincialCubes.length} provincial cubes found`
      );
    }
    // 50 ms delay before next metadata request, so max 20 requests per second, STATCAN limits 25 requests per ip address per second.
    await new Promise((r) => setTimeout(r, 50));
  }
  return provincialCubes;
}

// -----------------------------
// Precompute document frequencies for member names and dimension names
// across all cubes in the metadata set. Used by buildSearchText() to filter
// out generic cross-cutting terms before embedding.
//
// A member that appears in more than GENERIC_THRESHOLD of cubes (e.g. every
// NAICS industry name, standard age bands, ownership-type breakdowns) carries
// no subject-discriminating signal — embedding it adds noise, not meaning.
// IDF handles this naturally: df/N > threshold → IDF ≈ 0 → filter it out.
//
// Thresholds chosen from the real cubesMetadata.json distribution:
//   - 5% of cubes (180/3600) cleanly separates the 93 NAICS-industry members
//     that appear in 25%+ of cubes (pure structural breakdowns reused across
//     every subject area) from the 43,855 subject-specific members.
//   - Same 5% threshold for dimension names filters: Sex, Age group,
//     Statistics, NAICS, Gender, Characteristics — structural labels that
//     appear in 180–684 cubes — while keeping 2,157 specific dimension names.
// -----------------------------
const GENERIC_THRESHOLD_RATIO = 0.05; // >5% of all cubes → generic, filter out
const MEMBERS_PER_DIM = 8;            // max specific members to include per dimension

function buildDocumentFrequencies(cubes) {
  const memberDocFreq = new Map();
  const dimNameDocFreq = new Map();

  for (const cube of cubes) {
    const seenMembers = new Set();
    const seenDims = new Set();

    for (const dim of (cube.dimensions || [])) {
      const dimName = dim.dimensionNameEn;
      if (dimName) seenDims.add(dimName);
      for (const m of (dim.memberNames || [])) {
        seenMembers.add(m);
      }
    }

    for (const m of seenMembers) {
      memberDocFreq.set(m, (memberDocFreq.get(m) || 0) + 1);
    }
    for (const d of seenDims) {
      dimNameDocFreq.set(d, (dimNameDocFreq.get(d) || 0) + 1);
    }
  }

  return { memberDocFreq, dimNameDocFreq };
}

// -----------------------------
// Build the enriched search text for a single cube.
//
// Format: title. startDate. endDate. [dim name.] [member. member. ...]
// Only dimension names and member names that appear in ≤5% of cubes are
// included — those are the terms that actually distinguish this cube's
// subject matter from others. Generic structural labels (NAICS industries,
// age bands, Sex/Gender/Statistics dimensions) are dropped.
//
// This directly addresses the vocabulary gap between casual natural-language
// queries and formal StatCan table titles: "throws away the most garbage"
// has no overlap with "Disposal of waste, by source" at the title level, but
// the dimension members "Landfill", "Recycled", "Composted" give the
// embedding model something to anchor on.
// -----------------------------
function buildSearchText(cube, memberDocFreq, dimNameDocFreq, totalCubes) {
  const genericCutoff = Math.floor(totalCubes * GENERIC_THRESHOLD_RATIO);
  const parts = [cube.title];

  if (cube.startDate) parts.push(cube.startDate);
  if (cube.endDate) parts.push(cube.endDate);

  for (const dim of (cube.dimensions || [])) {
    const dimName = dim.dimensionNameEn;

    // Include the dimension name only if it's subject-specific (not generic).
    if (dimName && (dimNameDocFreq.get(dimName) || 0) <= genericCutoff) {
      parts.push(dimName);
    }

    // Include up to MEMBERS_PER_DIM specific members from this dimension.
    const specificMembers = (dim.memberNames || [])
      .filter(m => (memberDocFreq.get(m) || 0) <= genericCutoff)
      .slice(0, MEMBERS_PER_DIM);

    parts.push(...specificMembers);
  }

  return parts.join(". ");
}

// -----------------------------
// Generate embeddings for each cube using a single given model.
// Uses enriched search text (title + dates + specific dimension names +
// specific member names) instead of title + dates only.
// -----------------------------
async function addEmbeddings(cubes, modelName, memberDocFreq, dimNameDocFreq) {
  const totalCubes = cubes.length;
  console.log(`\nLoading embedding model (${modelName})...`);
  const extractor = await pipeline("feature-extraction", modelName);

  console.log(`Generating embeddings for ${cubes.length} cubes with ${modelName}...`);

  const results = [];

  for (let i = 0; i < cubes.length; i++) {
    const cube = cubes[i];
    const searchText = buildSearchText(cube, memberDocFreq, dimNameDocFreq, totalCubes);

    const embedding = await extractor(searchText, {
      pooling: "mean",
      normalize: true,
    });

    const roundedEmbedding = Array.from(embedding.data).map(
      (v) => Math.round(v * 1000) / 1000
    );

    results.push({
      cubeId: cube.cubeId,
      title: cube.title,
      startDate: cube.startDate,
      endDate: cube.endDate,
      frequencyCode: cube.frequencyCode,
      geographyCount: cube.geographyCount,
      embedding: roundedEmbedding,
    });

    if ((i + 1) % 20 === 0 || i + 1 === cubes.length) {
      console.log(`  Embedded ${i + 1}/${cubes.length}`);
    }
  }

  return results;
}

// -----------------------------
// Turn a model name like "Xenova/all-MiniLM-L6-v2" into a safe file slug
// like "all-MiniLM-L6-v2"
// -----------------------------
function modelToFileSlug(modelName) {
  return modelName.split("/").pop();
}

// -----------------------------
// Main
// -----------------------------
async function main() {
  console.log("StatCan Provincial Cube Index + Embeddings");
  console.log("===========================================\n");

  if (!(await testApi())) {
    console.error("\n API unavailable. Check network connectivity.");
    return;
  }

  const allCubes = await getAllCubes();
  if (!allCubes.length) {
    console.error("No cubes returned.");
    return;
  }

  const provincialCubes = await findProvincialCubes(allCubes);
  console.log(
    `\n=== Found ${provincialCubes.length} cubes with all 10 provinces ===`
  );

  if (!provincialCubes.length) {
    console.log("Nothing to embed. Exiting.");
    return;
  }

  // Save the full metadata once, independent of embeddings.
  await fs.writeFile(METADATA_OUT_PATH, JSON.stringify(provincialCubes, null, 2));
  const metaStats = await fs.stat(METADATA_OUT_PATH);
  console.log(
    `\n✓ Saved ${METADATA_OUT_PATH} (${(metaStats.size / (1024 * 1024)).toFixed(2)} MB)`
  );

  // Precompute document frequencies across all provincial cubes once — used
  // by buildSearchText() to filter generic dimension names and member names
  // before embedding. Done here rather than inside addEmbeddings() so it
  // runs once regardless of how many models are being embedded.
  console.log("\nComputing document frequencies for dimension/member filtering...");
  const { memberDocFreq, dimNameDocFreq } = buildDocumentFrequencies(provincialCubes);
  const genericCutoff = Math.floor(provincialCubes.length * GENERIC_THRESHOLD_RATIO);
  const genericMemberCount = [...memberDocFreq.values()].filter(n => n > genericCutoff).length;
  const genericDimCount = [...dimNameDocFreq.values()].filter(n => n > genericCutoff).length;
  console.log(`  ${memberDocFreq.size} distinct member names — ${genericMemberCount} generic (filtered), ${memberDocFreq.size - genericMemberCount} specific (kept)`);
  console.log(`  ${dimNameDocFreq.size} distinct dim names  — ${genericDimCount} generic (filtered), ${dimNameDocFreq.size - genericDimCount} specific (kept)`);

  // Log the first cube's enriched text so you can sanity-check what's being
  // embedded before waiting for the full run to complete.
  const sampleText = buildSearchText(provincialCubes[0], memberDocFreq, dimNameDocFreq, provincialCubes.length);
  console.log(`\nSample enriched text (first cube):\n  "${sampleText.slice(0, 300)}..."`);

  // Generate one embeddings file per model.
  for (const modelName of EMBEDDING_MODELS) {
    const cubesWithEmbeddings = await addEmbeddings(provincialCubes, modelName, memberDocFreq, dimNameDocFreq);

    const slug = modelToFileSlug(modelName);
    const outPath = `./cubesWithEmbeddings.${slug}.json`;
    await fs.writeFile(outPath, JSON.stringify(cubesWithEmbeddings, null, 2));

    const stats = await fs.stat(outPath);
    const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
    const dims = cubesWithEmbeddings[0]?.embedding.length ?? 0;

    console.log(`\n✓ Saved ${outPath}`);
    console.log(
      `  ${cubesWithEmbeddings.length} cubes · ${dims} dimensions · ${sizeMB} MB`
    );
  }

  console.log("\nAll done.");
}

main().catch(console.error);