// buildProvincialCubeIndex.js - WORKING VERSION with correct POST format
//
// Creates a JSON file containing ONLY cubes that:
// 1. Have a Geography dimension
// 2. Include ALL Canadian provinces
//

import axios from "axios";
import fs from "fs/promises";

const BASE_URL = "https://www150.statcan.gc.ca/t1/wds/rest";

// All provinces required for inclusion
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
  "British Columbia"
];

// -----------------------------
// Fetch all cubes
// -----------------------------
async function getAllCubes() {
  const url = `${BASE_URL}/getAllCubesListLite`;
  const response = await axios.get(url, {
    headers: {
      'Accept': 'application/json'
    }
  });
  console.log(`Fetched ${response.data?.length || 0} cubes total`);
  return response.data || [];
}

// -----------------------------
// Fetch cube metadata - CORRECT FORMAT
// POST body should be: [{"productId": 35100003}]
// -----------------------------
async function getCubeMetadata(productId) {
  const url = `${BASE_URL}/getCubeMetadata`;
  
  try {
    // Correct format from documentation: array of objects with productId
    const requestBody = [{ productId: productId }];
    
    const response = await axios({
      method: 'post',
      url: url,
      data: requestBody,
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      timeout: 15000
    });
    
    // Response should be an array where each element corresponds to a requested productId
    if (response.data && Array.isArray(response.data) && response.data.length > 0) {
      const result = response.data[0];
      
      // Check if we got a successful response
      if (result && result.status === "SUCCESS" && result.object) {
        return result.object;
      } else if (result && result.object) {
        return result.object;
      }
    }
    
    return null;
  } catch (err) {
    if (err.response?.status === 404 || err.response?.status === 406) {
      // This cube doesn't have metadata available
      return null;
    }
    console.error(`Failed metadata for ${productId}: ${err.message}`);
    return null;
  }
}

// -----------------------------
// Extract geography members from dimension array
// -----------------------------
function getGeographyMembers(metadata) {
  if (!metadata) return [];
  
  // According to docs, dimensions are in the 'dimension' array
  const dimensions = metadata.dimension;
  
  if (!dimensions || !Array.isArray(dimensions)) {
    return [];
  }

  // Find the Geography dimension
  const geographyDim = dimensions.find(dim => {
    const nameEn = dim.dimensionNameEn || '';
    return nameEn === "Geography" || nameEn.includes("Geography");
  });

  if (!geographyDim) {
    return [];
  }

  // Get members from the dimension
  const members = geographyDim.member;
  
  if (!members || !Array.isArray(members)) {
    return [];
  }

  // Extract member names
  const memberNames = members.map(member => {
    return member.memberNameEn || member.memberName;
  }).filter(Boolean);

  return memberNames;
}

// -----------------------------
// Check if cube contains all provinces
// -----------------------------
function containsAllProvinces(geographies) {
  if (!geographies.length) return false;
  
  const geographiesLower = geographies.map(g => g.toLowerCase());
  
  const missingProvinces = [];
  for (const province of REQUIRED_PROVINCES) {
    if (!geographiesLower.includes(province.toLowerCase())) {
      missingProvinces.push(province);
    }
  }
  
  return missingProvinces.length === 0;
}

// -----------------------------
// Test with known working cube from documentation
// -----------------------------
async function testWithDocumentationCube() {
  console.log("\n=== Testing with cube from documentation ===");
  console.log("Testing cube 35100003 (Correctional services data)...");
  
  const metadata = await getCubeMetadata(35100003);
  
  if (metadata) {
    console.log("  ✓ Successfully fetched metadata!");
    console.log(`  Title: ${metadata.cubeTitleEn}`);
    console.log(`  Product ID: ${metadata.productId}`);
    
    const geographies = getGeographyMembers(metadata);
    console.log(`  Geography members found: ${geographies.length}`);
    
    if (geographies.length > 0) {
      console.log(`  Sample geographies: ${geographies.slice(0, 5).join(', ')}`);
    }
    
    return true;
  } else {
    console.log("  ✗ Failed to fetch metadata");
    return false;
  }
}

// -----------------------------
// Search for provincial cubes
// -----------------------------
async function searchForProvincialCubes(cubes, maxToCheck = 200) {
  const provincialCubes = [];
  
  // Focus on cubes that likely have provincial data based on title
  const relevantCubes = cubes.filter(cube => {
    const title = (cube.cubeTitleEn || '').toLowerCase();
    const keywords = ['provincial', 'territorial', 'population', 'labour', 
                      'employment', 'income', 'economy', 'demographic'];
    return keywords.some(keyword => title.includes(keyword));
  });
  
  console.log(`\nFound ${relevantCubes.length} potentially relevant cubes`);
  console.log(`Checking first ${Math.min(relevantCubes.length, maxToCheck)} cubes...`);
  
  const BATCH_SIZE = 3;
  let processed = 0;
  
  for (let i = 0; i < Math.min(relevantCubes.length, maxToCheck); i += BATCH_SIZE) {
    const batch = relevantCubes.slice(i, i + BATCH_SIZE);
    
    const results = await Promise.all(
      batch.map(async (cube) => {
        const productId = cube.productId;
        
        const metadata = await getCubeMetadata(productId);
        
        if (!metadata) {
          return null;
        }
        
        const geographies = getGeographyMembers(metadata);
        
        if (geographies.length === 0) {
          return null;
        }
        
        const hasAllProvinces = containsAllProvinces(geographies);
        
        if (hasAllProvinces) {
          console.log(`\n✓ FOUND PROVINCIAL CUBE: ${productId}`);
          console.log(`  Title: ${cube.cubeTitleEn}`);
          console.log(`  Geography members: ${geographies.length}`);
          console.log(`  Start Date: ${metadata.cubeStartDate}`);
          console.log(`  End Date: ${metadata.cubeEndDate}`);
          
          return {
            cubeId: productId.toString(),
            title: cube.cubeTitleEn,
            startDate: metadata.cubeStartDate,
            endDate: metadata.cubeEndDate,
            releaseTime: metadata.releaseTime,
            frequencyCode: metadata.frequencyCode,
            geographyCount: geographies.length,
            surveyCode: metadata.surveyCode,
            subjectCode: metadata.subjectCode
          };
        }
        
        return null;
      })
    );
    
    const validResults = results.filter(Boolean);
    provincialCubes.push(...validResults);
    processed += batch.length;
    
    if (validResults.length > 0) {
      console.log(`\nFound ${validResults.length} new provincial cubes (total: ${provincialCubes.length})`);
    }
    
    if (processed % 30 === 0) {
      console.log(`Progress: ${processed}/${Math.min(relevantCubes.length, maxToCheck)} checked, found ${provincialCubes.length} provincial cubes`);
    }
    
    // Be respectful to the API
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  return provincialCubes;
}

// -----------------------------
// Main
// -----------------------------
async function main() {
  console.log("StatCan Provincial Cube Index Builder");
  console.log("=====================================\n");
  
  // Test with the cube from documentation to verify API works
  const apiWorks = await testWithDocumentationCube();
  
  if (!apiWorks) {
    console.error("\n❌ API test failed. Cannot proceed.");
    console.error("Please check network connectivity and API availability.");
    return;
  }
  
  console.log("\n✅ API is working correctly!\n");
  
  console.log("Fetching all cubes...");
  const cubes = await getAllCubes();
  
  if (!cubes || cubes.length === 0) {
    console.error("No cubes found!");
    return;
  }
  
  // Search for provincial cubes
  const provincialCubes = await searchForProvincialCubes(cubes, 9000); // Increase this number to check more cubes, but be mindful of rate limits
  
  console.log("\n=== RESULTS ===");
  console.log(`Found ${provincialCubes.length} cubes that contain all Canadian provinces`);
  
  if (provincialCubes.length > 0) {
    // Save the results
    await fs.writeFile(
      "./provincialCubeIndex.json",
      JSON.stringify(provincialCubes, null, 2)
    );
    console.log("\n✓ Saved provincialCubeIndex.json");
    
    // Save a summary for quick reference
    const summary = {
      generatedAt: new Date().toISOString(),
      totalProvincialCubes: provincialCubes.length,
      provincesRequired: REQUIRED_PROVINCES,
      cubes: provincialCubes.map(c => ({
        cubeId: c.cubeId,
        title: c.title,
        startDate: c.startDate,
        endDate: c.endDate
      }))
    };
    
    await fs.writeFile(
      "./provincialCubeIndex.summary.json",
      JSON.stringify(summary, null, 2)
    );
    console.log("✓ Saved provincialCubeIndex.summary.json");
    
    // Display the found cubes
    console.log("\nProvincial cubes found:");
    provincialCubes.forEach((cube, idx) => {
      console.log(`${idx + 1}. ${cube.cubeId}: ${cube.title.substring(0, 80)}`);
      console.log(`   Period: ${cube.startDate} to ${cube.endDate}`);
    });
  } else {
    console.log("\nNo provincial cubes found in the first 500 relevant cubes.");
    console.log("\nSuggestions:");
    console.log("1. Try increasing maxToCheck in searchForProvincialCubes");
    console.log("2. The geography dimension might use different province names");
    console.log("3. Some cubes might have provinces at a different level (e.g., 'Newfoundland' vs 'Newfoundland and Labrador')");
  }
}

main().catch(console.error);