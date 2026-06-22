import fs from "fs";
import { pipeline } from "@xenova/transformers";
import * as tf from "@tensorflow/tfjs";
import { fileURLToPath } from "url";
import path from "path";
import { loadMLP } from "./train_mlp.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rawData = JSON.parse(fs.readFileSync("benchmark_results.json", "utf8"));
const testData = JSON.parse(fs.readFileSync(path.join("input_data", "outputs", "test.json"), "utf8"));

const CATEGORIES = ["Government", "Income, pensions, spending and wealth", "International trade", "Health", "Labour", "Languages", "Manufacturing", "Population and demography", "Prices and price indexes", "Statistical methods", "Retail and wholesale", "Business and consumer services and culture", "Digital economy and society", "Transportation", "Travel and tourism", "Energy", "Science and technology", "Agriculture and food", "Business performance and ownership", "Construction", "Crime and justice", "Economic accounts", "Education, training and learning", "Environment", "Families, households and marital status", "Indigenous peoples", "Children and youth", "Immigration and ethnocultural diversity", "Older adults and population aging", "Society and community", "Housing"];
const labelToId = Object.fromEntries(CATEGORIES.map((c, i) => [c, i]));

async function evaluateMLP(modelId, embeddingDim) {
    const embedder = await pipeline("feature-extraction", modelId);
    const model = await loadMLP(modelId);

    const results = {}; 
    
    for (const item of testData) {
        if (!item.classification || !(item.classification in labelToId)) continue;
        
        const emb = await embedder(item.query, { pooling: "mean", normalize: true });
        const inputTensor = tf.tensor2d([Array.from(emb.data)], [1, embeddingDim]);
        
        const prediction = model.predict(inputTensor);
        const predIndex = prediction.argMax(1).dataSync()[0];
        const trueIndex = labelToId[item.classification];
        
        if (!results[item.classification]) results[item.classification] = { correct: 0, support: 0 };
        results[item.classification].support += 1;
        if (predIndex === trueIndex) results[item.classification].correct += 1;
        
        inputTensor.dispose();
        prediction.dispose();
    }
    
    embedder.dispose();
    model.dispose();
    
    const finalStats = {};
    for (const cat in results) {
        finalStats[cat] = {
            accuracy: results[cat].correct / results[cat].support,
            support: results[cat].support
        };
    }
    return finalStats;
}

function formatTable(categories, colWidths, dataAccessor) {
    let report = "Category".padEnd(colWidths.category);
    const modelNames = Object.keys(rawData);
    
    for (const model of modelNames) {
        report += ` | ${model.split('/').pop().padEnd(colWidths.model)}`;
    }
    report += "\n" + "-".repeat(report.length) + "\n";

    for (const category of categories) {
        let row = category.padEnd(colWidths.category);
        for (const model of modelNames) {
            const stats = dataAccessor(model, category);
            const cell = stats ? `${(stats.accuracy * 100).toFixed(0)}% (s:${stats.support})`.padEnd(colWidths.model) : "-".padEnd(colWidths.model);
            row += ` | ${cell}`;
        }
        report += row + "\n";
    }
    return report;
}

async function generateReports() {
    const categories = CATEGORIES.sort();
    const colWidths = { category: 45, model: 20 };
    const modelNames = Object.keys(rawData);

    let centroidReport = "=== Centroid Per-Category Accuracy ===\n\n";
    centroidReport += formatTable(categories, colWidths, (model, cat) => rawData[model].centroidPerCategory[cat]);
    fs.writeFileSync("centroid_results.txt", centroidReport);
    console.log("Saved centroid_results.txt");

    let mlpReport = "=== MLP Per-Category Accuracy ===\n\n";
    const mlpData = {};
    for (const modelId of modelNames) {
        mlpData[modelId] = await evaluateMLP(modelId, rawData[modelId].embeddingDim);
    }
    mlpReport += formatTable(categories, colWidths, (model, cat) => mlpData[model][cat]);
    fs.writeFileSync("mlp_results.txt", mlpReport);
    console.log("Saved mlp_results.txt");
}

generateReports().catch(console.error);