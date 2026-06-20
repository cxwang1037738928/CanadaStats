// classifier/train_centroids.js
import fs from "fs";

import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = path.join(__dirname, "query-classifier");
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// Turns a model id like "Xenova/all-mpnet-base-v2" into a filesystem-safe
// string, e.g. "Xenova_all-mpnet-base-v2".
export function safeModelKey(modelId) {
  return modelId.replace(/[\/\\]/g, "_");
}

export function trainCentroids(X, y, embeddingDim) {
  const sums = {};
  const counts = {};
  for (let i = 0; i < X.length; i++) {
    const label = y[i];
    if (!sums[label]) {
      sums[label] = new Array(embeddingDim).fill(0);
      counts[label] = 0;
    }
    counts[label]++;
    for (let j = 0; j < embeddingDim; j++) {
      sums[label][j] += X[i][j];
    }
  }
  const centroids = {};
  for (const label in sums) {
    const mean = sums[label].map(v => v / counts[label]);
    const norm = Math.sqrt(mean.reduce((s, x) => s + x * x, 0));
    centroids[label] = mean.map(x => x / norm);
  }
  return centroids;
}

// Filename is namespaced per embedding model (safeModelKey) so training
// multiple models in a loop doesn't have each one overwrite the last —
// previously this always wrote to the same centroids.json regardless of
// modelId, so only the last-trained model's centroids ever survived.
export function saveCentroids(centroids, modelId) {
  const key = safeModelKey(modelId);
  const outPath = path.join(OUTPUT_DIR, `centroids__${key}.json`);
  fs.writeFileSync(outPath, JSON.stringify(centroids, null, 2));
}

// Loads centroids previously saved by saveCentroids for the given modelId.
export function loadCentroids(modelId) {
  const key = safeModelKey(modelId);
  const inPath = path.join(OUTPUT_DIR, `centroids__${key}.json`);
  return JSON.parse(fs.readFileSync(inPath, "utf8"));
}