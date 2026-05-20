import axios from "axios";
import unzipper from "unzipper";
import csv from "csv-parser";
import fs from "fs/promises";

// StatsCan tables with metrics
const TABLES = [
  { metric: "median_after_tax_income", table: "11100135" },
  { metric: "low_income_rate", table: "11100017" },
  { metric: "employment_rate", table: "14100387" },
  { metric: "unemployment_rate", table: "14100287" },
  { metric: "life_expectancy", table: "13100394" },
  { metric: "post_secondary_education", table: "37100178" }
];

// Utility to find likely geography/year/value columns
function detectColumns(row) {
  const headers = Object.keys(row);

  const geoCol = headers.find(h =>
    /geo|province|region/i.test(h)
  );

  const yearCol = headers.find(h =>
    /ref_date|year/i.test(h)
  );

  const valueCol = headers.find(h =>
    /value/i.test(h)
  );

  if (!geoCol || !yearCol || !valueCol) {
    console.warn("Could not detect columns in row:", row);
  }

  return { geoCol, yearCol, valueCol };
}

// Fetch CSV and parse
async function fetchAndParse(tableId) {
  const url = `https://www150.statcan.gc.ca/n1/en/tbl/csv/${tableId}-eng.zip`;
  const response = await axios({ method: "GET", url, responseType: "stream" });

  return new Promise((resolve, reject) => {
    const rows = [];
    response.data
      .pipe(unzipper.ParseOne())
      .pipe(csv())
      .on("data", row => rows.push(row))
      .on("end", () => resolve(rows))
      .on("error", reject);
  });
}

async function main() {
  const merged = {};
  const allMetrics = TABLES.map(t => t.metric);

  for (const dataset of TABLES) {
    try {
      console.log(`Fetching ${dataset.metric} from table ${dataset.table}`);
      const rows = await fetchAndParse(dataset.table);

      if (!rows.length) continue;

      // Detect columns from first row
      const { geoCol, yearCol, valueCol } = detectColumns(rows[0]);
      if (!geoCol || !yearCol || !valueCol) continue;

      for (const row of rows) {
        const province = row[geoCol];
        const year = Number(row[yearCol]);
        const value = Number(row[valueCol]);

        if (!province || !year || isNaN(value)) continue;

        const key = `${province}|${year}`;
        if (!merged[key]) {
          merged[key] = { province, year };
          allMetrics.forEach(metric => (merged[key][metric] = null));
        }

        merged[key][dataset.metric] = value;
      }
    } catch (err) {
      console.warn(`Failed to fetch table ${dataset.table}: ${err.message}`);
    }
  }

  const mergedArray = Object.values(merged);

  // Creates data directory at src/data/canada_metrics_full.json
  await fs.mkdir("../data", { recursive: true });

  await fs.writeFile(
    "../data/canada_metrics_full.json",
    JSON.stringify(mergedArray, null, 2)
  );

  console.log(`Merged dataset created with ${mergedArray.length} rows (province-year).`);
}

main().catch(console.error);