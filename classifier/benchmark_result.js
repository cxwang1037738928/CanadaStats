// classifier/benchmark_result.js
import fs from "fs";
import { ChartJSNodeCanvas } from 'chartjs-node-canvas';

const rawData = JSON.parse(fs.readFileSync("benchmark_results.json", "utf8"));
const metrics = ['accuracy', 'val_accuracy', 'test_accuracy'];

console.log("=== Final Accuracy Comparison (Terminal Summary) ===");
console.log("Model".padEnd(30) + "| Final Test | Centroid");
console.log("-".repeat(50));
for (const [model, stats] of Object.entries(rawData)) {
    const final = stats.epochHistory[stats.epochHistory.length - 1];
    console.log(`${model.padEnd(29)}| ${(final.test_accuracy * 100).toFixed(1)}%     | ${(stats.centroidAcc * 100).toFixed(1)}%`);
}

// Colors are assigned once per model and reused across every chart so a given
// model is visually consistent (e.g. same hue in the line charts and the dot plot).
function colorForModel(index, total) {
    return `hsl(${Math.round((index * 360) / total)}, 70%, 50%)`;
}

async function generateCharts() {
    const width = 800;
    const height = 400;
    const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height });

    const modelNames = Object.keys(rawData);

    for (const metric of metrics) {
        const datasets = Object.entries(rawData).map(([model, stats], index) => ({
            label: model,
            data: stats.epochHistory.map(h => h[metric]),
            borderColor: colorForModel(index, modelNames.length),
            fill: false,
            tension: 0.1
        }));
        // Add Centroid Baseline as a dashed line for reference
        datasets.push({
            label: 'Centroid Baseline (Avg)',
            data: [], // Note: Centroid is constant, handled in tooltips/legend
            borderColor: '#555',
            borderDash: [5, 5],
            borderWidth: 2
        });
        const config = {
            type: 'line',
            data: {
                labels: Array.from({ length: rawData[modelNames[0]].epochHistory.length }, (_, i) => i + 1),
                datasets: datasets
            },
            options: {
                plugins: {
                    title: { display: true, text: `Model Performance: ${metric.toUpperCase()}` },
                    subtitle: { display: true, text: 'Dashed line represents representative baseline (see terminal for exact values)' }
                }
            }
        };
        const image = await chartJSNodeCanvas.renderToBuffer(config);
        fs.writeFileSync(`performance_${metric}.png`, image);
        console.log(`Saved performance_${metric}.png`);
    }

    await generateCentroidScatterChart(chartJSNodeCanvas, modelNames);
}

// New: scatter/dot plot of per-category centroid accuracy, one color per model.
// X-axis = category (categorical), Y-axis = accuracy (0-1). Each (model, category)
// pair contributes one dot.
async function generateCentroidScatterChart(chartJSNodeCanvas, modelNames) {
    // Build the full ordered category list from whichever model has the most
    // complete breakdown (they should all share the same category set).
    const categorySet = new Set();
    for (const model of modelNames) {
        const perCat = rawData[model].centroidPerCategory || {};
        Object.keys(perCat).forEach(cat => categorySet.add(cat));
    }
    const categories = Array.from(categorySet);
    const categoryIndex = Object.fromEntries(categories.map((c, i) => [c, i]));

    const datasets = modelNames.map((model, index) => {
        const perCat = rawData[model].centroidPerCategory || {};
        const color = colorForModel(index, modelNames.length);
        const points = categories
            .filter(cat => perCat[cat] !== undefined)
            .map(cat => ({
                x: categoryIndex[cat],
                y: perCat[cat].accuracy
            }));
        return {
            label: model,
            data: points,
            backgroundColor: color,
            borderColor: color,
            pointRadius: 5,
            pointHoverRadius: 7,
            showLine: false
        };
    });

    const config = {
        type: 'scatter',
        data: { datasets },
        options: {
            plugins: {
                title: { display: true, text: 'Centroid Accuracy per Category per Model' },
                legend: { display: true, position: 'top' }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: -0.5,
                    max: categories.length - 0.5,
                    title: { display: true, text: 'Category' },
                    ticks: {
                        stepSize: 1,
                        callback: (value) => categories[value] ?? ''
                    }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'Centroid Accuracy' },
                    ticks: {
                        callback: (value) => `${Math.round(value * 100)}%`
                    }
                }
            }
        }
    };

    // Wider canvas since there are many categories along the x-axis.
    const wideChartJSNodeCanvas = new ChartJSNodeCanvas({ width: 1400, height: 700 });
    const image = await wideChartJSNodeCanvas.renderToBuffer(config);
    fs.writeFileSync('centroid_accuracy_per_category.png', image);
    console.log('Saved centroid_accuracy_per_category.png');
}

generateCharts().catch(console.error);