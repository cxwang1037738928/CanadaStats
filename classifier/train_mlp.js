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

// Turns a model id like "Xenova/all-mpnet-base-v2" into a filesystem-safe
// string, e.g. "Xenova_all-mpnet-base-v2". Used to namespace saved files per
// embedding model so training multiple models in a loop doesn't overwrite
// each other's saved MLP.
export function safeModelKey(modelId) {
  return modelId.replace(/[\/\\]/g, "_");
}

export async function trainAndSaveMLP(modelId, trainX, trainY, valX, valY, testX, testY, embeddingDim, numCats) {
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

  // Manual save using fs to bypass the 'file://' protocol handler issue (no
  // tfjs-node available). IMPORTANT: model.toJSON() does NOT return valid
  // model topology — it returns a JSON *string* of internal serialization
  // state that is not the same shape tf.loadLayersModel expects, and is not
  // safely re-stringifiable. The correct topology object only comes from
  // model.save() via a custom IOHandler (tf.io.withSaveHandler), which is
  // the same mechanism @tensorflow/tfjs-node's file:// handler uses
  // internally — we're just intercepting the artifacts instead of writing
  // them to a single file ourselves.
  let artifacts;
  await model.save(
    tf.io.withSaveHandler(async (modelArtifacts) => {
      artifacts = modelArtifacts;
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
    })
  );

  // Filenames are namespaced per embedding model (safeModelKey) so training
  // multiple models in a loop doesn't have each one overwrite the last.
  const key = safeModelKey(modelId);
  fs.writeFileSync(
    path.join(OUTPUT_DIR, `mlp-model__${key}.json`),
    JSON.stringify(artifacts.modelTopology, null, 2)
  );
  fs.writeFileSync(
    path.join(OUTPUT_DIR, `mlp-weights__${key}.json`),
    JSON.stringify(
      {
        weightSpecs: artifacts.weightSpecs,
        weightDataBase64: Buffer.from(artifacts.weightData).toString("base64")
      },
      null,
      2
    )
  );
  
  // Cleanup
  [trainXs, trainYs, valXs, valYs, testXs, testYs].forEach(t => t.dispose());
  model.dispose();
  
  return history;
}

// Loads a model previously saved by trainAndSaveMLP for the given modelId,
// reading the same two namespaced files back via tf.io.fromMemory — the
// load-side counterpart to tf.io.withSaveHandler used above. This avoids
// tf.loadLayersModel('file://...'), which requires @tensorflow/tfjs-node
// and is not available in this project.
export async function loadMLP(modelId) {
  const key = safeModelKey(modelId);
  const modelTopology = JSON.parse(
    fs.readFileSync(path.join(OUTPUT_DIR, `mlp-model__${key}.json`), "utf8")
  );
  const { weightSpecs, weightDataBase64 } = JSON.parse(
    fs.readFileSync(path.join(OUTPUT_DIR, `mlp-weights__${key}.json`), "utf8")
  );
  const weightData = Uint8Array.from(Buffer.from(weightDataBase64, "base64")).buffer;

  return tf.loadLayersModel(
    tf.io.fromMemory({ modelTopology, weightSpecs, weightData })
  );
}