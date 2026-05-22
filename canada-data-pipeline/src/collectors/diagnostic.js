// compactDiagnostic.js

import axios from "axios";

const BASE_URL = "https://www150.statcan.gc.ca/t1/wds/rest";
const TEST_CUBE_ID = 35100003;

// -----------------------------------
// Fetch metadata
// -----------------------------------
async function getCubeMetadata(productId) {

  const response = await axios.post(
    `${BASE_URL}/getCubeMetadata`,
    [{ productId }],
    {
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json"
      }
    }
  );

  return response.data?.[0]?.object || null;
}

// -----------------------------------
// Pretty-print dimensions
// -----------------------------------
function printCompact(cube) {

  console.log(`\n${cube.cubeTitleEn}`);
  console.log(`Cube: ${cube.productId}`);
  console.log(`Period: ${cube.cubeStartDate} → ${cube.cubeEndDate}`);

  console.log("\nDIMENSIONS");
  console.log("==========");

  cube.dimension.forEach((dim, i) => {

    console.log(
      `\n[${i + 1}] ${dim.dimensionNameEn} ` +
      `(${dim.member?.length || 0} members)`
    );

    const members = dim.member || [];

    members.forEach(member => {

      const id = member.memberId;
      const parent = member.parentMemberId;

      let line =
        `  ${id}. ${member.memberNameEn}`;

      if (parent) {
        line += ` ← parent:${parent}`;
      }

      if (member.vectorId) {
        line += ` | vector:${member.vectorId}`;
      }

      console.log(line);
    });
  });
}

// -----------------------------------
// Main
// -----------------------------------
async function main() {

  const cube = await getCubeMetadata(TEST_CUBE_ID);

  if (!cube) {
    console.log("No cube metadata found.");
    return;
  }

  printCompact(cube);
}

main().catch(console.error);