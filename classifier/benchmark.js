import fs from "fs";
import 'dotenv/config';

const SERPER_API_KEY = process.env.SERPER_API_KEY;
const QUERY_FILE = "query.json";
const OUTPUT_FILE = "searchResults.json";

const MAX_PAGES = 5;
const RESULTS_PER_PAGE = 10;

/**
 * Returns true if the URL is from the StatCan domain AND ends with
 * 'pid=' followed only by digits (i.e. reversed, the first non-digit
 * characters spell '=dip').
 *
 * Examples that match:
 *   https://www23.statcan.gc.ca/imdb/p2SV.pl?Function=getSurvey&pid=98765
 *   https://www150.statcan.gc.ca/t1/tbl1/en/dtbl!/pid=9810028402
 */
function isStatCanPidUrl(url) {
  if (!url.toLowerCase().includes("statcan.gc.ca")) return false;

  const noFragment = url.split("#")[0];
  const reversed = noFragment.split("").reverse().join("");

  // When reversed, the URL must start with digits then =dip
  return /^\d*=dip/i.test(reversed);
}

async function searchSerper(query, page = 1) {
  const response = await fetch("https://google.serper.dev/search", {
    method: "POST",
    headers: {
      "X-API-KEY": SERPER_API_KEY,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ q: query, num: RESULTS_PER_PAGE, page }),
  });

  if (!response.ok) {
    console.error(`HTTP error from Serper API: ${response.status} ${response.statusText}`);
    return [];
  }

  const data = await response.json();
  return data.organic ?? [];
}

async function main() {
  // Read query
  if (!fs.existsSync(QUERY_FILE)) {
    console.error(`Error: '${QUERY_FILE}' not found.`);
    process.exit(1);
  }

  const query = fs.readFileSync(QUERY_FILE, "utf-8").trim();
  if (!query) {
    console.error("Error: query.txt is empty.");
    process.exit(1);
  }

  console.log(`Query: ${query}`);
  console.log("Searching for StatCan links ending in 'pid=<number>'...\n");

  const matchingLinks = [];

  for (let page = 1; page <= MAX_PAGES; page++) {
    console.log(`Fetching page ${page}...`);
    const results = await searchSerper(query, page);

    if (!results.length) {
      console.log("No more results returned.");
      break;
    }

    for (const result of results) {
      const link = result.link ?? "";
      if (isStatCanPidUrl(link)) {
        console.log(`  ✓ Match found: ${link}`);
        matchingLinks.push(link);
      }
    }

    // Stop after first page that yields a match.
    // Remove this block if you want ALL matches across all pages.
    if (matchingLinks.length) break;
  }

  // Write results
  if (matchingLinks.length) {
    fs.writeFileSync(OUTPUT_FILE, matchingLinks.join("\n") + "\n", "utf-8");
    console.log(`\n${matchingLinks.length} link(s) written to '${OUTPUT_FILE}'.`);
  } else {
    fs.writeFileSync(OUTPUT_FILE, "No matching StatCan pid= links found.\n", "utf-8");
    console.log("\nNo matching links found. See searchResults.txt.");
  }
}

main().catch((err) => {
  console.error("Unexpected error:", err);
  process.exit(1);
});