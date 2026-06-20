// classifier/train_centroids.js
import fs from "fs";

import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = path.join(__dirname, "query-classifier");
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

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

export function saveCentroids(centroids, modelId) {
  const path = `${OUTPUT_DIR}/centroids.json`; // Specific path requested
  fs.writeFileSync(path, JSON.stringify(centroids, null, 2));
}