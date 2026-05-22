// loadProvincialData.js
// Loads provincial cube data from provincialCubeIndex.json into PostgreSQL using Prisma

import { PrismaClient } from '@prisma/client';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const prisma = new PrismaClient();

// Provincial cube data structure from our index
// Each entry has:
// {
//   cubeId: string,
//   title: string,
//   startDate: string,
//   endDate: string,
//   releaseTime: string,
//   frequencyCode: number,
//   geographyCount: number,
//   surveyCode: array,
//   subjectCode: array
// }

// Helper to extract year from date string (YYYY-MM-DD)
function extractYear(dateString) {
  if (!dateString) return null;
  const match = dateString.match(/^(\d{4})/);
  return match ? parseInt(match[1]) : null;
}

// Map StatCan frequency codes to our metric keys
function getMetricKeyFromCube(cube) {
  const title = cube.title.toLowerCase();
  
  // Define pattern matching for different metrics
  const metricPatterns = [
    { pattern: /population/i, metricKey: 'population', displayName: 'Population' },
    { pattern: /employment|labour force|job/i, metricKey: 'employment', displayName: 'Employment' },
    { pattern: /unemployment/i, metricKey: 'unemployment_rate', displayName: 'Unemployment Rate' },
    { pattern: /income|earnings|salary/i, metricKey: 'income', displayName: 'Income' },
    { pattern: /housing|dwelling|home/i, metricKey: 'housing', displayName: 'Housing' },
    { pattern: /education|school|student/i, metricKey: 'education', displayName: 'Education' },
    { pattern: /health|hospital|medical/i, metricKey: 'health', displayName: 'Health' },
    { pattern: /crime|criminal|correctional/i, metricKey: 'crime', displayName: 'Crime Statistics' },
    { pattern: /immigration|migrant/i, metricKey: 'immigration', displayName: 'Immigration' },
    { pattern: /economy|gdp|economic/i, metricKey: 'economy', displayName: 'Economy' }
  ];
  
  for (const { pattern, metricKey, displayName } of metricPatterns) {
    if (pattern.test(title)) {
      return { metricKey, displayName };
    }
  }
  
  // Default fallback
  return { metricKey: 'other', displayName: 'Other Statistics' };
}

// Create or get metric record
async function getOrCreateMetric(metricKey, displayName, source = 'Statistics Canada') {
  return await prisma.metric.upsert({
    where: { key: metricKey },
    update: {},
    create: {
      key: metricKey,
      displayName: displayName,
      source: source
    }
  });
}

// Create or get province record
async function getOrCreateProvince(provinceName) {
  // Map full province names to codes
  const provinceMap = {
    'Newfoundland and Labrador': 'NL',
    'Prince Edward Island': 'PE',
    'Nova Scotia': 'NS',
    'New Brunswick': 'NB',
    'Quebec': 'QC',
    'Ontario': 'ON',
    'Manitoba': 'MB',
    'Saskatchewan': 'SK',
    'Alberta': 'AB',
    'British Columbia': 'BC',
    'Yukon': 'YT',
    'Northwest Territories': 'NT',
    'Nunavut': 'NU'
  };
  
  const code = provinceMap[provinceName];
  if (!code) {
    console.warn(`Unknown province: ${provinceName}`);
    return null;
  }
  
  return await prisma.province.upsert({
    where: { code: code },
    update: {},
    create: {
      code: code,
      name: provinceName
    }
  });
}

// For now, we're just indexing cubes, not loading actual data
// This script will create a registry of available data cubes
async function registerProvincialCubes(cubes) {
  console.log(`Registering ${cubes.length} provincial cubes...`);
  
  const registeredCubes = [];
  
  for (const cube of cubes) {
    const { metricKey, displayName } = getMetricKeyFromCube(cube);
    
    // Get or create the metric
    const metric = await getOrCreateMetric(metricKey, displayName);
    
    // Store cube metadata as a JSON field (we'll need to add this to schema)
    // For now, we'll store in a separate table or as a JSON field
    
    registeredCubes.push({
      cubeId: cube.cubeId,
      title: cube.title,
      metricId: metric.id,
      metricKey: metricKey,
      startYear: extractYear(cube.startDate),
      endYear: extractYear(cube.endDate),
      frequencyCode: cube.frequencyCode,
      geographyCount: cube.geographyCount
    });
  }
  
  return registeredCubes;
}

// Main function to load the index
async function loadProvincialCubeIndex() {
  try {
    console.log('Loading provincialCubeIndex.json...');
    
    const dataPath = path.join(__dirname, 'provincialCubeIndex.json');
    const fileExists = await fs.access(dataPath).then(() => true).catch(() => false);
    
    if (!fileExists) {
      console.error('provincialCubeIndex.json not found. Please run buildProvincialCubeIndex.js first.');
      return;
    }
    
    const fileContent = await fs.readFile(dataPath, 'utf-8');
    const cubes = JSON.parse(fileContent);
    
    console.log(`Found ${cubes.length} provincial cubes to register`);
    
    // Register cubes in database
    const registered = await registerProvincialCubes(cubes);
    
    console.log('\n=== Registration Complete ===');
    console.log(`Registered ${registered.length} provincial cubes`);
    
    // Group by metric type
    const byMetric = registered.reduce((acc, cube) => {
      acc[cube.metricKey] = (acc[cube.metricKey] || 0) + 1;
      return acc;
    }, {});
    
    console.log('\nCubes by metric type:');
    Object.entries(byMetric).forEach(([metric, count]) => {
      console.log(`  ${metric}: ${count} cubes`);
    });
    
    // Save registered cubes summary
    const summaryPath = path.join(__dirname, 'registeredCubes.json');
    await fs.writeFile(summaryPath, JSON.stringify(registered, null, 2));
    console.log(`\nSaved registration summary to ${summaryPath}`);
    
  } catch (error) {
    console.error('Error loading provincial cube index:', error);
  } finally {
    await prisma.$disconnect();
  }
}

// Run the loader
loadProvincialCubeIndex();