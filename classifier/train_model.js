import fs from "fs";
import * as tf from "@tensorflow/tfjs";
import { pipeline } from "@xenova/transformers";

const CATEGORIES = [
  "Government",
  "Income, pensions, spending and wealth",
  "International trade",
  "Health",
  "Labour",
  "Languages",
  "Manufacturing",
  "Population and demography",
  "Prices and price indexes",
  "Statistical methods",
  "Retail and wholesale",
  "Business and consumer services and culture",
  "Digital economy and society",
  "Transportation",
  "Travel and tourism",
  "Energy",
  "Science and technology",
  "Agriculture and food",
  "Business performance and ownership",
  "Construction",
  "Crime and justice",
  "Economic accounts",
  "Education, training and learning",
  "Environment",
  "Families, households and marital status",
  "Indigenous peoples",
  "Children and youth",
  "Immigration and ethnocultural diversity",
  "Older adults and population aging",
  "Society and community",
  "Housing"
];

const labelToId = {};
const idToLabel = {};

CATEGORIES.forEach((c, i) => {
  labelToId[c] = i;
  idToLabel[i] = c;
});

function shuffle(arrX, arrY) {
  const idx = [...arrX.keys()];
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return {
    X: idx.map(i => arrX[i]),
    y: idx.map(i => arrY[i])
  };
}

function cosine(a, b) {
  let dot = 0, ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    ma += a[i] * a[i];
    mb += b[i] * b[i];
  }
  return dot / (Math.sqrt(ma) * Math.sqrt(mb));
}

function trainCentroids(X, y) {
  const sums = {};
  const counts = {};

  for (let i = 0; i < X.length; i++) {
    const label = y[i];
    if (!sums[label]) {
      sums[label] = new Array(384).fill(0);
      counts[label] = 0;
    }
    counts[label]++;
    for (let j = 0; j < 384; j++) {
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

function printCategoryStats(y, title) {
  const counts = {};
  for (const label of y) counts[label] = (counts[label] || 0) + 1;
  console.log(`\n=== ${title} Category Distribution ===`);
  for (let i = 0; i < CATEGORIES.length; i++) {
    console.log(`${CATEGORIES[i]}: ${counts[i] || 0}`);
  }
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
  console.log(`\n=== Centroid Accuracy per Category ===`);
  for (let i = 0; i < CATEGORIES.length; i++) {
    const acc = (correct[i] || 0) / (total[i] || 1);
    console.log(`${CATEGORIES[i]}: ${(acc * 100).toFixed(2)}%`);
  }
}

async function main() {
  console.log("Loading data...");
  const inputs = JSON.parse(fs.readFileSync("inputs.json", "utf8"));
  const classifications = JSON.parse(fs.readFileSync("classifications.json", "utf8"));

  const classMap = {};
  for (const item of classifications) {
    classMap[String(item.id)] = item.classification;
  }

  console.log("Loading embedding model...");
  const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

  const X = [];
  const y = [];
  let processed = 0;

  for (const item of inputs) {
    const category = classMap[String(item.id)];
    if (!category || !(category in labelToId)) continue;

    const emb = await embedder(item.query, { pooling: "mean", normalize: true });
    X.push(Array.from(emb.data));
    y.push(labelToId[category]);

    processed++;
    if (processed % 200 === 0) console.log(`  Embedded ${processed}/${inputs.length}`);
  }

  console.log(`\nLoaded ${X.length} samples`);
  printCategoryStats(y, "FULL DATASET");

  const shuffled = shuffle(X, y);
  const n = shuffled.X.length;
  const trainEnd = Math.floor(n * 0.7);
  const valEnd   = Math.floor(n * 0.85);

  const trainX = shuffled.X.slice(0, trainEnd);
  const trainY = shuffled.y.slice(0, trainEnd);
  const valX   = shuffled.X.slice(trainEnd, valEnd);
  const valY   = shuffled.y.slice(trainEnd, valEnd);
  const testX  = shuffled.X.slice(valEnd);
  const testY  = shuffled.y.slice(valEnd);

  console.log(`\nSplit → Train=${trainX.length} Val=${valX.length} Test=${testX.length}`);
  printCategoryStats(trainY, "TRAIN");
  printCategoryStats(testY,  "TEST");

  // ── Centroid baseline ────────────────────────────────────────────────────────
  console.log("\nTraining centroids...");
  const centroids = trainCentroids(trainX, trainY);
  perCategoryCentroidAccuracy(centroids, testX, testY);
  const centroidAcc = evaluateCentroids(centroids, testX, testY);
  console.log(`\nOverall Centroid Accuracy: ${(centroidAcc * 100).toFixed(2)}%`);

  // ── MLP ──────────────────────────────────────────────────────────────────────
  console.log("\nTraining MLP...");

  // FIX: sparseCategoricalCrossentropy requires float32 labels, NOT int32
  const trainXs = tf.tensor2d(trainX, [trainX.length, 384], "float32");
  const trainYs = tf.tensor1d(trainY, "float32");   // <── was "int32", caused the crash

  const valXs = tf.tensor2d(valX, [valX.length, 384], "float32");
  const valYs = tf.tensor1d(valY, "float32");       // <── same fix

  const testXs = tf.tensor2d(testX, [testX.length, 384], "float32");
  const testYs = tf.tensor1d(testY, "float32");     // <── same fix

  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [384], units: 128, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: CATEGORIES.length, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"]
  });

  model.summary();

  await model.fit(trainXs, trainYs, {
    epochs: 30,
    batchSize: 32,
    validationData: [valXs, valYs],
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: "val_loss", patience: 5 })
    ]
  });

  // ── Evaluate ─────────────────────────────────────────────────────────────────
  const [, accTensor] = model.evaluate(testXs, testYs);
  const mlpAcc = (await accTensor.data())[0];

  console.log(`\nMLP Test Accuracy: ${(mlpAcc * 100).toFixed(2)}%`);
  console.log(`Improvement over centroid: ${((mlpAcc - centroidAcc) * 100).toFixed(2)} pts`);

  // ── Save artifacts ───────────────────────────────────────────────────────────
  fs.mkdirSync("./query-classifier", { recursive: true });

  // Save MLP weights manually — model.save("file://...") requires @tensorflow/tfjs-node
  // which isn't installed. Instead we serialize the architecture + weights by hand.
  const modelTopology = model.toJSON(null, false);

  const weightData = [];
  const weightSpecs = [];

  for (const layer of model.layers) {
    for (const weight of layer.getWeights()) {
      weightSpecs.push({
        name:  weight.name,
        dtype: weight.dtype,
        shape: weight.shape
      });
      weightData.push(Array.from(weight.dataSync()));
    }
  }

  fs.writeFileSync(
    "./query-classifier/mlp-model.json",
    JSON.stringify({ modelTopology, weightSpecs, weightData }, null, 2)
  );

  fs.writeFileSync(
    "./query-classifier/centroids.json",
    JSON.stringify(centroids, null, 2)
  );

  fs.writeFileSync(
    "./query-classifier/metadata.json",
    JSON.stringify({
      categories:     CATEGORIES,
      embeddingModel: "Xenova/all-MiniLM-L6-v2",
      centroidAccuracy: centroidAcc,
      mlpAccuracy:      mlpAcc
    }, null, 2)
  );

  // Clean up tensors
  trainXs.dispose(); trainYs.dispose();
  valXs.dispose();   valYs.dispose();
  testXs.dispose();  testYs.dispose();
  accTensor.dispose();

  console.log("\nArtifacts saved to ./query-classifier/");
  console.log("Done.");
}

main().catch(console.error);