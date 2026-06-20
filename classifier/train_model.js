// classifier/train_model.js
import fs from "fs";
import path from "path";
import { pipeline } from "@xenova/transformers";
import { trainCentroids, saveCentroids } from "./train_centroids.js";
import { trainAndSaveMLP } from "./train_mlp.js";

const MODELS = ["Xenova/all-MiniLM-L6-v2", "Xenova/all-MiniLM-L12-v2", "Xenova/paraphrase-MiniLM-L3-v2", "Xenova/all-mpnet-base-v2", "Xenova/bge-small-en-v1.5"];
const CATEGORIES = ["Government", "Income, pensions, spending and wealth", "International trade", "Health", "Labour", "Languages", "Manufacturing", "Population and demography", "Prices and price indexes", "Statistical methods", "Retail and wholesale", "Business and consumer services and culture", "Digital economy and society", "Transportation", "Travel and tourism", "Energy", "Science and technology", "Agriculture and food", "Business performance and ownership", "Construction", "Crime and justice", "Economic accounts", "Education, training and learning", "Environment", "Families, households and marital status", "Indigenous peoples", "Children and youth", "Immigration and ethnocultural diversity", "Older adults and population aging", "Society and community", "Housing"];

const labelToId = Object.fromEntries(CATEGORIES.map((c, i) => [c, i]));

// epochHistory rows look like { epoch, accuracy, val_accuracy, test_accuracy }.
// CSV split label -> epochHistory field name.
const SPLIT_FIELDS = {
  train: "accuracy",
  val: "val_accuracy",
  test: "test_accuracy"
};

function escapeCsvField(value) {
  const str = String(value);
  return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
}

// Converts { [modelId]: { epochHistory: [...] } } into long-format CSV rows:
// model, split, epoch, accuracy — one row per (model, split, epoch).
function buildEpochAccuracyCsv(benchmarkResults) {
  const header = ["model", "split", "epoch", "accuracy"];
  const rows = [header];

  for (const [modelId, stats] of Object.entries(benchmarkResults)) {
    for (const point of stats.epochHistory) {
      for (const [split, field] of Object.entries(SPLIT_FIELDS)) {
        const acc = point[field];
        if (acc === undefined || acc === null) continue;
        rows.push([modelId, split, point.epoch, acc]);
      }
    }
  }

  return rows.map(row => row.map(escapeCsvField).join(",")).join("\n") + "\n";
}

function cosine(a, b) {
  let dot = 0, ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; ma += a[i] * a[i]; mb += b[i] * b[i]; }
  return dot / (Math.sqrt(ma) * Math.sqrt(mb));
}

function evaluateCentroids(centroids, X, y) {
  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    let best = -Infinity, pred = null;
    for (const label in centroids) {
      const score = cosine(X[i], centroids[label]);
      if (score > best) { best = score; pred = Number(label); }
    }
    if (pred === y[i]) correct++;
  }
  return correct / X.length;
}

function perCategoryCentroidAccuracy(centroids, X, y) {
  const correct = {}, total = {};
  for (let i = 0; i < X.length; i++) {
    const trueLabel = y[i];
    let best = -Infinity, pred = null;
    for (const label in centroids) {
      const score = cosine(X[i], centroids[label]);
      if (score > best) { best = score; pred = Number(label); }
    }
    total[trueLabel] = (total[trueLabel] || 0) + 1;
    if (pred === trueLabel) correct[trueLabel] = (correct[trueLabel] || 0) + 1;
  }
  return CATEGORIES.reduce((acc, cat, i) => {
    acc[cat] = { accuracy: (correct[i] || 0) / (total[i] || 1), support: total[i] || 0 };
    return acc;
  }, {});
}

// Data now lives pre-split on disk under input_data/outputs/{train,val,test}/,
// one JSON array per split, each item shaped { id, query, classification }.
// No more inputs.json/classifications.json lookup-by-id, and no more in-code
// 70/15/15 shuffle-split — that's all handled by however the splits were built.
const DATA_DIR = path.join("input_data", "outputs");

function loadSplit(splitName) {
  const filePath = path.join(DATA_DIR, `${splitName}.json`);
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

// Embeds one split's items with the given embedder, keeping only items whose
// classification matches a known category.
async function embedSplit(embedder, items) {
  const X = [], y = [];
  for (const item of items) {
    const cat = item.classification;
    if (cat && cat in labelToId) {
      const emb = await embedder(item.query, { pooling: "mean", normalize: true });
      X.push(Array.from(emb.data));
      y.push(labelToId[cat]);
    }
  }
  return { X, y };
}

async function main() {
  const trainItems = loadSplit("train");
  const valItems = loadSplit("val");
  const testItems = loadSplit("test");

  const benchmarkResults = {};

  for (const modelId of MODELS) {

    console.log(`\n==================================================`);
    console.log(`STARTING TRAINING FOR MODEL: ${modelId}`);
    console.log(`==================================================`);
    const embedder = await pipeline("feature-extraction", modelId);

    const { X: trainX, y: trainY } = await embedSplit(embedder, trainItems);
    const { X: valX, y: valY } = await embedSplit(embedder, valItems);
    const { X: testX, y: testY } = await embedSplit(embedder, testItems);

    await embedder.dispose();

    if (trainX.length === 0) {
      console.log(`No training data for ${modelId}, skipping.`);
      continue;
    }

    const embeddingDim = trainX[0].length;

    const centroids = trainCentroids(trainX, trainY, embeddingDim);
    saveCentroids(centroids, modelId);

    const epochHistory = await trainAndSaveMLP(modelId, trainX, trainY, valX, valY, testX, testY, embeddingDim, CATEGORIES.length);

    benchmarkResults[modelId] = {
      embeddingDim,
      centroidAcc: evaluateCentroids(centroids, testX, testY),
      centroidPerCategory: perCategoryCentroidAccuracy(centroids, testX, testY),
      epochHistory
    };
  }
  fs.writeFileSync("benchmark_results.json", JSON.stringify(benchmarkResults, null, 2));
  console.log("Saved benchmark_results.json");

  const csv = buildEpochAccuracyCsv(benchmarkResults);
  fs.writeFileSync("epoch_accuracy.csv", csv);
  console.log("Saved epoch_accuracy.csv");
}
main().catch(console.error);