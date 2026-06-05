// buildProvincialCubesWithEmbeddings.js
//
// 1. Fetches all StatCan cubes
// 2. Filters to only cubes that include all 10 Canadian provinces in the geography
// 3. Generates embeddings using Xenova/all-MiniLM-L6-v2
// 4. Saves results to cubesWithEmbeddings.json
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
// Extract geography member names from metadata
// -----------------------------
function getGeographyMembers(metadata) {
  if (!metadata) return [];

  const dimensions = metadata.dimension;
  if (!Array.isArray(dimensions)) return [];

  const geographyDim = dimensions.find((dim) => {
    const name = dim.dimensionNameEn || "";
    return name === "Geography" || name.includes("Geography");
  });

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
// Verify API connectivity with a known cube
// -----------------------------
async function testApi() {
  console.log("Testing API with cube 35100003...");
  const metadata = await getCubeMetadata(35100003);
  if (metadata) {
    console.log(`  ✓ API OK — "${metadata.cubeTitleEn}"`);
    return true;
  }
  console.log("  ✗ API test failed");
  return false;
}

// -----------------------------
// Find all provincial cubes
// -----------------------------
async function findProvincialCubes(cubes, maxToCheck = 9000) {
  const limit = Math.min(cubes.length, maxToCheck);
  const provincialCubes = [];

  for (let i = 0; i < limit; i++) {
    const cube = relevantCubes[i];

    const metadata = await getCubeMetadata(cube.productId);

    if (metadata) {
      const geographies = getGeographyMembers(metadata);

      if (containsAllProvinces(geographies)) {
        console.log(`\n✓ ${cube.productId}: ${cube.cubeTitleEn}`);

        provincialCubes.push({
          cubeId: cube.productId.toString(),
          title: cube.cubeTitleEn,
          startDate: metadata.cubeStartDate,
          endDate: metadata.cubeEndDate,
          releaseTime: metadata.releaseTime,
          frequencyCode: metadata.frequencyCode,
          geographyCount: geographies.length,
          surveyCode: metadata.surveyCode,
          subjectCode: metadata.subjectCode,
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
// Generate embeddings for each cube
// -----------------------------
async function addEmbeddings(cubes) {
  console.log("\nLoading embedding model (Xenova/all-MiniLM-L6-v2)...");
  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  console.log(`Generating embeddings for ${cubes.length} cubes...`);

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
// Main
// -----------------------------
async function main() {
  console.log("StatCan Provincial Cube Index + Embeddings");
  console.log("===========================================\n");

  if (!(await testApi())) {
    console.error("\n❌ API unavailable. Check network connectivity.");
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

  const cubesWithEmbeddings = await addEmbeddings(provincialCubes);

  const outPath = "./cubesWithEmbeddings.json";
  await fs.writeFile(outPath, JSON.stringify(cubesWithEmbeddings, null, 2));

  const stats = await fs.stat(outPath);
  const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
  const dims = cubesWithEmbeddings[0]?.embedding.length ?? 0;

  console.log(`\n✓ Saved ${outPath}`);
  console.log(`  ${cubesWithEmbeddings.length} cubes · ${dims} dimensions · ${sizeMB} MB`);
}

main().catch(console.error);