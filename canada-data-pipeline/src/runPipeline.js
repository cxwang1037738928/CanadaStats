import { fetchRawData } from "./collectors/fetchRawData.js";

async function main() {
  console.log("Starting pipeline");

  await collectUnemployment();

  console.log("Pipeline complete");
}

main().catch(console.error);