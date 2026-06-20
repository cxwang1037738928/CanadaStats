// classifier/train_model.js
import fs from "fs";
import { pipeline } from "@xenova/transformers";
import { trainCentroids, saveCentroids } from "./train_centroids.js";
import { trainAndSaveMLP } from "./train_mlp.js";

const MODELS = ["Xenova/all-MiniLM-L6-v2", "Xenova/all-MiniLM-L12-v2", "Xenova/paraphrase-MiniLM-L3-v2", "Xenova/all-mpnet-base-v2", "Xenova/bge-small-en-v1.5"];
const CATEGORIES = ["Government", "Income, pensions, spending and wealth", "International trade", "Health", "Labour", "Languages", "Manufacturing", "Population and demography", "Prices and price indexes", "Statistical methods", "Retail and wholesale", "Business and consumer services and culture", "Digital economy and society", "Transportation", "Travel and tourism", "Energy", "Science and technology", "Agriculture and food", "Business performance and ownership", "Construction", "Crime and justice", "Economic accounts", "Education, training and learning", "Environment", "Families, households and marital status", "Indigenous peoples", "Children and youth", "Immigration and ethnocultural diversity", "Older adults and population aging", "Society and community", "Housing"];

const labelToId = Object.fromEntries(CATEGORIES.map((c, i) => [c, i]));

function shuffle(arrX, arrY) {
  const idx = [...arrX.keys()].sort(() => Math.random() - 0.5);
  return { X: idx.map(i => arrX[i]), y: idx.map(i => arrY[i]) };
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

async function main() {
  const inputs = JSON.parse(fs.readFileSync("inputs.json", "utf8"));
  const classifications = JSON.parse(fs.readFileSync("classifications.json", "utf8"));
  const benchmarkResults = {};

  for (const modelId of MODELS) {

    console.log(`\n==================================================`);
    console.log(`STARTING TRAINING FOR MODEL: ${modelId}`);
    console.log(`==================================================`);
    const embedder = await pipeline("feature-extraction", modelId);
    const X = [], y = [];
    for (const item of inputs) {
      const cat = classifications.find(c => String(c.id) === String(item.id))?.classification;
      if (cat && cat in labelToId) {
        const emb = await embedder(item.query, { pooling: "mean", normalize: true });
        X.push(Array.from(emb.data));
        y.push(labelToId[cat]);
      }
    }
    await embedder.dispose();
    const embeddingDim = X[0].length;
    const { X: sX, y: sY } = shuffle(X, y);
    const trainEnd = Math.floor(sX.length * 0.7);
    const valEnd = Math.floor(sX.length * 0.85);

    const centroids = trainCentroids(sX.slice(0, trainEnd), sY.slice(0, trainEnd), embeddingDim);
    saveCentroids(centroids, modelId);
    
    const epochHistory = await trainAndSaveMLP(sX.slice(0, trainEnd), sY.slice(0, trainEnd), sX.slice(trainEnd, valEnd), sY.slice(trainEnd, valEnd), sX.slice(valEnd), sY.slice(valEnd), embeddingDim, CATEGORIES.length);

    benchmarkResults[modelId] = {
      embeddingDim,
      centroidAcc: evaluateCentroids(centroids, sX.slice(valEnd), sY.slice(valEnd)),
      centroidPerCategory: perCategoryCentroidAccuracy(centroids, sX.slice(valEnd), sY.slice(valEnd)),
      epochHistory
    };
  }
  fs.writeFileSync("benchmark_results.json", JSON.stringify(benchmarkResults, null, 2));
}
main().catch(console.error);