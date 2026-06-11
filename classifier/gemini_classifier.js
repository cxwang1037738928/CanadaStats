import fs from "fs";
import 'dotenv/config';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.0-flash";

const TITLES_FILE = "inputs.json"; // file containing both titles and user queries to classify, each being id'ed
const OUTPUT_FILE = "classifications.json"; // resulting classifications from Gemini, each containing an id and a classification field

const CHUNK_SIZE = 500;
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;

/**
 * Builds the prompt sent to Gemini for a given chunk of titles.
 * Edit this to suit your classification needs.
 */
function buildPrompt(titles, chunkIndex, totalChunks) {
  return `You are a classification assistant. Below is a list of titles (chunk ${chunkIndex + 1} of ${totalChunks}).
Classify each title and return your results in this exact format, one per line:
<title> | <classification>

Titles:
${titles.map((t, i) => `${i + 1}. ${t}`).join("\n")}`;
}

async function classifyChunk(titles, chunkIndex, totalChunks) {
  const prompt = buildPrompt(titles, chunkIndex, totalChunks);

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
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  return text.trim();
}

async function main() {
  // Read titles file
  if (!fs.existsSync(TITLES_FILE)) {
    console.error(`Error: '${TITLES_FILE}' not found.`);
    process.exit(1);
  }

  const allLines = fs
    .readFileSync(TITLES_FILE, "utf-8")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (!allLines.length) {
    console.error("Error: titles.txt is empty.");
    process.exit(1);
  }

  // Split into chunks of CHUNK_SIZE
  const chunks = [];
  for (let i = 0; i < allLines.length; i += CHUNK_SIZE) {
    chunks.push(allLines.slice(i, i + CHUNK_SIZE));
  }

  console.log(`Loaded ${allLines.length} titles → ${chunks.length} chunk(s) of up to ${CHUNK_SIZE} lines.\n`);

  // Clear / create output file
  fs.writeFileSync(OUTPUT_FILE, "", "utf-8");

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const start = i * CHUNK_SIZE + 1;
    const end = start + chunk.length - 1;
    console.log(`Processing chunk ${i + 1}/${chunks.length} (lines ${start}–${end})...`);

    try {
      const result = await classifyChunk(chunk, i, chunks.length);

      // Append chunk header + results to the output file
      const header = `\n--- Chunk ${i + 1}/${chunks.length} (lines ${start}–${end}) ---\n`;
      fs.appendFileSync(OUTPUT_FILE, header + result + "\n", "utf-8");

      console.log(`  ✓ Chunk ${i + 1} written.`);
    } catch (err) {
      console.error(`  ✗ Chunk ${i + 1} failed: ${err.message}`);
      fs.appendFileSync(
        OUTPUT_FILE,
        `\n--- Chunk ${i + 1} ERROR ---\n${err.message}\n`,
        "utf-8"
      );
    }
  }

  console.log(`\nDone. All classifications written to '${OUTPUT_FILE}'.`);
}

main().catch((err) => {
  console.error("Unexpected error:", err);
  process.exit(1);
});