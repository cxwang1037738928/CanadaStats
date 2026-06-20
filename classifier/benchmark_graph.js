import fs from "fs";
import { ChartJSNodeCanvas } from 'chartjs-node-canvas';
import { fileURLToPath } from "url";
import path from "path";

const rawData = JSON.parse(fs.readFileSync("benchmark_results.json", "utf8"));
const metrics = ['accuracy', 'val_accuracy', 'test_accuracy'];

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
        datasets.push({
            label: 'Centroid Baseline (Avg)',
            data: [],
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
                    title: { display: true, text: `Model Performance: ${metric.toUpperCase()}` }
                }
            }
        };
        const image = await chartJSNodeCanvas.renderToBuffer(config);
        fs.writeFileSync(`performance_${metric}.png`, image);
        console.log(`Saved performance_${metric}.png`);
    }

    await generateCentroidScatterChart(chartJSNodeCanvas, modelNames);
}

async function generateCentroidScatterChart(chartJSNodeCanvas, modelNames) {
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
            plugins: { title: { display: true, text: 'Centroid Accuracy per Category per Model' } },
            scales: {
                x: {
                    type: 'linear',
                    ticks: { callback: (value) => categories[value] ?? '' }
                },
                y: { min: 0, max: 1 }
            }
        }
    };

    const wideChartJSNodeCanvas = new ChartJSNodeCanvas({ width: 1400, height: 700 });
    const image = await wideChartJSNodeCanvas.renderToBuffer(config);
    fs.writeFileSync('centroid_accuracy_per_category.png', image);
    console.log('Saved centroid_accuracy_per_category.png');
}

generateCharts().catch(console.error);