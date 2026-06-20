import fs from "fs";
import 'dotenv/config';

const SERPER_API_KEY = process.env.SERPER_API_KEY;
const QUERY_FILE = "query.json";
const OUTPUT_FILE = "searchResults.json";

// Hardcoded to 10 to guarantee exactly 1 token/credit per API call
const RESULTS_PER_PAGE = 10; 

/**
 * Returns true if the URL is from the StatCan domain AND ends with
 * 'pid=' followed only by digits (i.e. reversed, the first non-digit
 * characters spell '=dip').
 */
function isStatCanPidUrl(url) {
  if (!url.toLowerCase().includes("statcan.gc.ca")) return false;
  const noFragment = url.split("#")[0];
  const reversed = noFragment.split("").reverse().join("");
  return /^\d*=dip/i.test(reversed);
}

async function searchSerper(query) {
  const response = await fetch("https://google.serper.dev/search", {
    method: "POST",
    headers: {
      "X-API-KEY": SERPER_API_KEY,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ 
      q: query, 
      num: RESULTS_PER_PAGE, // Strictly limited to 10 results
      page: 1                // Always stick to page 1
    }),
  });

  if (!response.ok) {
    console.error(`HTTP error from Serper API: ${response.status} ${response.statusText}`);
    return [];
  }

  const data = await response.json();
  return data.organic ?? [];
}

async function main() {
  if (!fs.existsSync(QUERY_FILE)) {
    console.error(`Error: '${QUERY_FILE}' not found.`);
    process.exit(1);
  }

  const raw = JSON.parse(fs.readFileSync(QUERY_FILE, "utf-8"));
  const queries = Array.isArray(raw) ? raw : [raw];

  if (!queries.length) {
    console.error("Error: query.json is empty.");
    process.exit(1);
  }

  const allResults = []; 

  for (const { id, query } of queries) {
    if (!query?.trim()) {
      console.warn(`  Skipping entry id=${id} — missing query text.`);
      continue;
    }

    console.log(`\nQuery [id=${id}]: ${query}`);
    console.log("Searching Page 1 (Top 10 results)...");

    const modified_query = `${query} StatCan`;
    const results = await searchSerper(modified_query);
    let foundForQuery = false;

    for (const result of results) {
      const link = result.link ?? "";
      if (isStatCanPidUrl(link)) {
        console.log(`  ✓ Match found: ${link}`);
        allResults.push({ id, query, link });
        foundForQuery = true;
      }
    }

    if (!foundForQuery) {
      console.log(`  No matching link found in the top 10 results for id=${id}.`);
      allResults.push({ id, query, link: null });
    }
  }

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allResults, null, 2), "utf-8");
  console.log(`\n${allResults.filter(r => r.link).length} match(es) written to '${OUTPUT_FILE}'.`);
}

main().catch((err) => {
  console.error("Unexpected error:", err);
  process.exit(1);
});