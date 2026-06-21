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
// Generate embeddings for each cube using a single given model.
// Embedding text is title + start/end date only, as before.
// -----------------------------
async function addEmbeddings(cubes, modelName) {
  console.log(`\nLoading embedding model (${modelName})...`);
  const extractor = await pipeline("feature-extraction", modelName);

  console.log(`Generating embeddings for ${cubes.length} cubes with ${modelName}...`);

  const results = [];

  for (let i = 0; i < cubes.length; i++) {
    const cube = cubes[i];
    const searchText = `${cube.title} ${cube.startDate || ""} ${cube.endDate || ""}`;

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

  // Generate one embeddings file per model.
  for (const modelName of EMBEDDING_MODELS) {
    const cubesWithEmbeddings = await addEmbeddings(provincialCubes, modelName);

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