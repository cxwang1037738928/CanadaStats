// classifier/train_mlp.js
import * as tf from "@tensorflow/tfjs";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = path.join(__dirname, "query-classifier");

// Ensure the directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

export async function trainAndSaveMLP(trainX, trainY, valX, valY, testX, testY, embeddingDim, numCats) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [embeddingDim], units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: numCats, activation: "softmax" }));
  model.compile({ optimizer: "adam", loss: "sparseCategoricalCrossentropy", metrics: ["accuracy"] });

  const history = [];
  const trainXs = tf.tensor2d(trainX, [trainX.length, embeddingDim], "float32");
  const trainYs = tf.tensor1d(trainY, "float32");
  const valXs = tf.tensor2d(valX, [valX.length, embeddingDim], "float32");
  const valYs = tf.tensor1d(valY, "float32");
  const testXs = tf.tensor2d(testX, [testX.length, embeddingDim], "float32");
  const testYs = tf.tensor1d(testY, "float32");

  await model.fit(trainXs, trainYs, {
    epochs: 30,
    validationData: [valXs, valYs],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const evalRes = model.evaluate(testXs, testYs);
        const testAcc = (await evalRes[1].data())[0];
        history.push({ epoch, accuracy: logs.acc, val_accuracy: logs.val_acc, test_accuracy: testAcc });
        evalRes.forEach(t => t.dispose());
      }
    }
  });

  // Manual save using fs to bypass the 'file://' protocol handler issue
  const modelJson = model.toJSON();
  const weights = await model.getWeights();
  const weightData = await Promise.all(weights.map(async (w) => ({
    name: w.name,
    shape: w.shape,
    data: Array.from(await w.data())
  })));

  fs.writeFileSync(path.join(OUTPUT_DIR, "mlp-model.json"), JSON.stringify(modelJson, null, 2));
  fs.writeFileSync(path.join(OUTPUT_DIR, "mlp-weights.json"), JSON.stringify(weightData, null, 2));
  
  // Cleanup
  [trainXs, trainYs, valXs, valYs, testXs, testYs].forEach(t => t.dispose());
  model.dispose();
  
  return history;
}