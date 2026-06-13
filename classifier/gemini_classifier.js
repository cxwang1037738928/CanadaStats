import fs from "fs";
import 'dotenv/config';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.0-flash";
const TITLES_FILE = "inputs.json";
const OUTPUT_FILE = "classifications.json";
const CHUNK_SIZE = 500;
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;

/**
 * Prompt asks Gemini to return one JSON object per line so we can
 * reliably parse id + classification without regex fragility.
 */
function buildPrompt(items, chunkIndex, totalChunks) {
  return `You are a classification assistant. Below is a list of titles (chunk ${chunkIndex + 1} of ${totalChunks}).

Classify each title. Return ONLY a JSON array — no markdown, no explanation — where each element has exactly two fields:
  "id"             – the id provided for that title (copy it exactly)
  "classification" – your classification label for the title

Example output format:
[
  {"id": 1, "classification": "Economics & Finance"},
  {"id": 2, "classification": "Health & Medicine"}
]

Titles:
${items.map(({ id, title }) => `id=${id}: ${title}`).join("\n")}`;
}

/**
 * Call Gemini and parse the returned JSON array.
 * Returns an array of { id, classification } objects.
 */
async function classifyChunk(items, chunkIndex, totalChunks) {
  const prompt = buildPrompt(items, chunkIndex, totalChunks);

  const response = await fetch(GEMINI_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.2 },
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Gemini API error (${response.status}): ${errText}`);
  }

  const data = await response.json();
  const raw = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";

  // Strip possible markdown code fences before parsing
  const cleaned = raw.replace(/^```json\s*/i, "").replace(/```\s*$/, "").trim();

  try {
    const parsed = JSON.parse(cleaned);
    if (!Array.isArray(parsed)) throw new Error("Response was not a JSON array.");
    return parsed.map(({ id, classification }) => ({ id, classification }));
  } catch (err) {
    throw new Error(`Failed to parse Gemini response as JSON: ${err.message}\nRaw:\n${raw}`);
  }
}

async function main() {
  if (!fs.existsSync(TITLES_FILE)) {
    console.error(`Error: '${TITLES_FILE}' not found.`);
    process.exit(1);
  }

  // inputs.json: [{ "id": 1, "title": "..." }, ...]
  const allItems = JSON.parse(fs.readFileSync(TITLES_FILE, "utf-8"));

  if (!Array.isArray(allItems) || !allItems.length) {
    console.error("Error: inputs.json is empty or not an array.");
    process.exit(1);
  }

  // Filter out any entries missing id or title
  const validItems = allItems.filter(({ id, title }) => {
    if (id == null || !title?.trim()) {
      console.warn(`  Skipping entry id=${id} — missing id or title.`);
      return false;
    }
    return true;
  });

  // Split into chunks
  const chunks = [];
  for (let i = 0; i < validItems.length; i += CHUNK_SIZE) {
    chunks.push(validItems.slice(i, i + CHUNK_SIZE));
  }

  console.log(`Loaded ${validItems.length} titles → ${chunks.length} chunk(s) of up to ${CHUNK_SIZE}.\n`);

  const allClassifications = []; // accumulates { id, classification } across all chunks

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const start = i * CHUNK_SIZE + 1;
    const end = start + chunk.length - 1;
    console.log(`Processing chunk ${i + 1}/${chunks.length} (items ${start}–${end})...`);

    try {
      const results = await classifyChunk(chunk, i, chunks.length);
      allClassifications.push(...results);
      console.log(`  ✓ Chunk ${i + 1}: ${results.length} classification(s) received.`);
    } catch (err) {
      console.error(`  ✗ Chunk ${i + 1} failed: ${err.message}`);
      // Push error placeholders so ids aren't silently dropped
      for (const { id } of chunk) {
        allClassifications.push({ id, classification: null, error: err.message });
      }
    }
  }

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allClassifications, null, 2), "utf-8");
  const good = allClassifications.filter(r => r.classification !== null).length;
  console.log(`\nDone. ${good}/${allClassifications.length} classifications written to '${OUTPUT_FILE}'.`);
}

main().catch((err) => {
  console.error("Unexpected error:", err);
  process.exit(1);
});