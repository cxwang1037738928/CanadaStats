// src/tests/testSearch.js
import fs from 'fs/promises';
import path from 'path';
import axios from 'axios';
import { pipeline } from '@xenova/transformers';

async function fetchDataByCoordinate(productId, coordinate, latestN = 5) {
  try {
    const url = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromCubePidCoordAndLatestNPeriods";
    const requestBody = [{
      productId: parseInt(productId),
      coordinate: coordinate,
      latestN: latestN
    }];
    
    const response = await axios.post(url, requestBody, {
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      timeout: 10000
    });
    
    if (response.data && response.data[0] && response.data[0].object) {
      return response.data[0].object;
    }
    return null;
  } catch (error) {
    return null;
  }
}

async function fetchMultiDimensionalData(cubeId, cubeTitle, userQuery) {
  // Get metadata
  const url = "https://www150.statcan.gc.ca/t1/wds/rest/getCubeMetadata";
  const response = await axios.post(url, [{ productId: parseInt(cubeId) }], {
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    timeout: 10000
  });
  
  if (!response.data || !response.data[0] || !response.data[0].object) {
    return { success: false, error: 'Could not fetch metadata' };
  }
  
  const metadata = response.data[0].object;
  
  // Find Geography dimension
  const geoDimension = metadata.dimension.find(dim => 
    dim.dimensionNameEn === "Geography" || dim.dimensionNameEn?.includes("Geography")
  );
  
  if (!geoDimension) {
    return { success: false, error: 'No geography dimension found' };
  }
  
  // Map province names
  const provinceMapping = {
    'Newfoundland and Labrador': 'Newfoundland and Labrador',
    'Prince Edward Island': 'Prince Edward Island',
    'Nova Scotia': 'Nova Scotia',
    'New Brunswick': 'New Brunswick',
    'Quebec': 'Quebec',
    'Ontario': 'Ontario',
    'Manitoba': 'Manitoba',
    'Saskatchewan': 'Saskatchewan',
    'Alberta': 'Alberta',
    'British Columbia': 'British Columbia',
    'Yukon': 'Yukon',
    'Northwest Territories': 'Northwest Territories',
    'Nunavut': 'Nunavut'
  };
  
  // Get all provinces
  const provinces = [];
  for (const member of geoDimension.member) {
    for (const [statcanName, standardName] of Object.entries(provinceMapping)) {
      if (member.memberNameEn && member.memberNameEn === statcanName) {
        provinces.push({
          id: member.memberId,
          name: standardName,
          originalName: member.memberNameEn
        });
        break;
      }
    }
  }
  
  // Identify other dimensions
  const otherDimensions = metadata.dimension.filter(dim => 
    dim.dimensionNameEn !== "Geography" && !dim.dimensionNameEn?.includes("Geography")
  );
  
  // Find the dimension that contains our target metric based on user query
  const queryLower = userQuery.toLowerCase();
  let targetMetricDimension = null;
  let targetMetricValue = null;
  
  for (const dim of otherDimensions) {
    if (dim.member) {
      for (const member of dim.member) {
        const memberName = member.memberNameEn?.toLowerCase() || '';
        if (queryLower.includes('unemployment') && memberName.includes('unemployment')) {
          targetMetricDimension = dim;
          targetMetricValue = member;
          break;
        } else if (queryLower.includes('employment') && memberName.includes('employment') && !memberName.includes('unemployment')) {
          targetMetricDimension = dim;
          targetMetricValue = member;
          break;
        } else if (queryLower.includes('population') && memberName.includes('population')) {
          targetMetricDimension = dim;
          targetMetricValue = member;
          break;
        } else if (queryLower.includes('gini') && memberName.includes('gini')) {
          targetMetricDimension = dim;
          targetMetricValue = member;
          break;
        } else if (queryLower.includes('median') && memberName.includes('median')) {
          targetMetricDimension = dim;
          targetMetricValue = member;
          break;
        }
      }
      if (targetMetricDimension) break;
    }
  }
  
  // Build dimensions structure
  const dimensions = [
    { name: "Province", values: provinces.map(p => p.name) },
    ...otherDimensions.map(dim => ({
      name: dim.dimensionNameEn,
      values: dim.member?.filter(m => m.memberId !== 0).map(m => m.memberNameEn) || []
    }))
  ];
  
  const dataPoints = [];
  
  // For each province
  for (const province of provinces) {
    const baseCoordinate = Array(10).fill('0');
    baseCoordinate[0] = province.id.toString();
    
    async function exploreDimension(dimIndex, currentCoordinate, currentSelections) {
      if (dimIndex >= otherDimensions.length) {
        const data = await fetchDataByCoordinate(cubeId, currentCoordinate.join('.'), 5);
        
        if (data && data.vectorDataPoint && data.vectorDataPoint.length > 0) {
          const point = data.vectorDataPoint[0];
          let value = point.value;
          const decimals = point.decimals || 0;
          const year = point.refPer ? point.refPer.split('-')[0] : 'N/A';
          
          if (decimals > 0) {
            value = value / Math.pow(10, decimals);
          }
          
          dataPoints.push({
            dimensions: { Province: province.name, ...currentSelections },
            value: value,
            year: year
          });
        }
        return;
      }
      
      const dim = otherDimensions[dimIndex];
      const dimName = dim.dimensionNameEn;
      
      // If this dimension contains our target metric and we found it, use that specific value
      let membersToExplore = [];
      if (targetMetricDimension && dim.dimensionNameEn === targetMetricDimension.dimensionNameEn) {
        membersToExplore = [targetMetricValue];
      } else {
        // Otherwise, find "Total" or "All" or "Both sexes"
        const totalMember = dim.member?.find(m => 
          m.memberNameEn?.toLowerCase().includes('total') || 
          m.memberNameEn?.toLowerCase().includes('all') ||
          m.memberNameEn?.toLowerCase().includes('both sexes') ||
          m.memberNameEn?.toLowerCase().includes('both genders') ||
          m.memberNameEn === 'Both sexes' ||
          m.memberNameEn === 'Total'
        );
        
        if (totalMember) {
          membersToExplore = [totalMember];
        } else if (dim.member && dim.member.length > 0) {
          const firstMember = dim.member.find(m => m.memberId !== 0);
          if (firstMember) {
            membersToExplore = [firstMember];
          }
        }
      }
      
      for (const member of membersToExplore) {
        currentCoordinate[dimIndex + 1] = member.memberId.toString();
        await exploreDimension(dimIndex + 1, currentCoordinate, {
          ...currentSelections,
          [dimName]: member.memberNameEn
        });
      }
    }
    
    await exploreDimension(0, [...baseCoordinate], {});
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  // Organize data
  const multiDimData = {
    cubeId: cubeId,
    title: cubeTitle,
    dimensions: dimensions,
    data: dataPoints,
    targetMetric: targetMetricValue ? targetMetricValue.memberNameEn : null,
    metadata: {
      startDate: metadata.cubeStartDate,
      endDate: metadata.cubeEndDate,
      footnotes: metadata.footnote?.slice(0, 3).map(f => f.footnotesEn) || []
    }
  };
  
  return { success: true, data: multiDimData };
}

function formatValueWithContext(value, metricName) {
  const metricLower = metricName?.toLowerCase() || '';
  
  if (metricLower.includes('rate') || metricLower.includes('unemployment') || metricLower.includes('participation')) {
    return value.toFixed(1) + '%';
  } else if (metricLower.includes('gini')) {
    return value.toFixed(3);
  } else if (metricLower.includes('median') || metricLower.includes('income')) {
    return '$' + value.toLocaleString();
  } else if (metricLower.includes('population')) {
    if (value > 1000) {
      return (value / 1000).toFixed(1) + 'K';
    }
    return value.toLocaleString();
  } else {
    return value.toLocaleString();
  }
}

function displayMultiDimData(multiDimData, userQuery) {
  console.log(`\n  📊 Target metric: ${multiDimData.targetMetric || 'Auto-detected'}`);
  console.log(`\n  📐 Dimensions in this cube:`);
  multiDimData.dimensions.forEach((dim, idx) => {
    console.log(`     [${idx}] ${dim.name}: ${dim.values.length} unique values`);
    if (dim.values.length <= 5) {
      console.log(`         ${dim.values.join(', ')}`);
    } else {
      console.log(`         ${dim.values.slice(0, 5).join(', ')}... (${dim.values.length} total)`);
    }
  });
  
  // Filter to show only relevant data points
  const relevantData = multiDimData.data.filter(point => 
    !multiDimData.targetMetric || 
    Object.values(point.dimensions).some(v => v === multiDimData.targetMetric)
  );
  
  console.log(`\n  📊 Data points for ${multiDimData.targetMetric || 'requested metric'}:`);
  console.log('  ' + '-'.repeat(80));
  
  relevantData.forEach(point => {
    const formattedValue = formatValueWithContext(point.value, multiDimData.targetMetric);
    console.log(`     ${point.dimensions.Province.padEnd(25)} ${formattedValue.padEnd(15)} (${point.year})`);
  });
  
  // Create list-of-lists representation
  console.log(`\n  📋 List-of-lists representation:`);
  console.log(`     dataAsListOfLists = [`);

  const headers = ['Province', 'Value', 'Year'];
  console.log(`       ${JSON.stringify(headers)},`);
  
  relevantData.forEach((point, idx) => {
    const formattedValue = formatValueWithContext(point.value, multiDimData.targetMetric);
    const row = [point.dimensions.Province, formattedValue, point.year];
    console.log(`       ${JSON.stringify(row)}${idx === relevantData.length - 1 ? '' : ','}`);
  });
  
  console.log(`     ]`);
  
  if (multiDimData.metadata.footnotes.length > 0) {
    console.log(`\n  📝 Notes: ${multiDimData.metadata.footnotes[0].substring(0, 200)}...`);
  }
}

async function testSearch() {
  const embeddingsPath = path.join(process.cwd(), '../collectors/cubesWithEmbeddings.json');
  console.log(`Loading embeddings from: ${embeddingsPath}`);
  const embeddingsData = await fs.readFile(embeddingsPath, 'utf8');
  const cubes = JSON.parse(embeddingsData);
  
  const userQuery = "unemployment rate of youths";
  
  console.log(`\nQuery: "${userQuery}"`);
  console.log(`Total cubes: ${cubes.length}\n`);
  
  console.log('Finding best cube...');
  const start = Date.now();
  const results = await semanticSearch(userQuery, cubes, 10);
  
  let bestMultiDimData = null;
  
  for (let i = 0; i < Math.min(results.length, 5); i++) {
    const cube = results[i];
    console.log(`\n${'='.repeat(80)}`);
    console.log(`Testing cube ${i + 1}: ${cube.cubeId}`);
    console.log(`Title: ${cube.title.substring(0, 80)}...`);
    console.log(`Similarity: ${(cube.similarity * 100).toFixed(1)}%`);
    
    console.log(`\nFetching data...`);
    const result = await fetchMultiDimensionalData(cube.cubeId, cube.title, userQuery);
    
    if (result.success && result.data.data.length > 0) {
      console.log(`\n✅ Success!`);
      displayMultiDimData(result.data, userQuery);
      
      if (!bestMultiDimData && result.data.targetMetric) {
        bestMultiDimData = result.data;
      }
      break;
    } else {
      console.log(`   ⚠ ${result.error || 'No data retrieved'}`);
    }
  }
  
  console.log(`\n${'='.repeat(80)}`);
  console.log(`Search completed in ${Date.now() - start}ms`);
  
  if (bestMultiDimData) {
    console.log(`\n✅ Best cube for your query: ${bestMultiDimData.cubeId}`);
    console.log(`   Metric: ${bestMultiDimData.targetMetric}`);
    console.log(`\n   Ready for Canada map visualization!`);
  }
}

async function semanticSearch(query, cubes, topK = 5) {
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  
  const queryEmbedding = await extractor(query, { 
    pooling: 'mean', 
    normalize: true 
  });
  
  const queryVector = Array.from(queryEmbedding.data).map(v => Math.round(v * 1000) / 1000);
  
  function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
    }
    return dotProduct;
  }
  
  const results = cubes.map(cube => ({
    cubeId: cube.cubeId,
    title: cube.title,
    similarity: cosineSimilarity(queryVector, cube.embedding)
  }));
  
  results.sort((a, b) => b.similarity - a.similarity);
  
  return results.slice(0, topK);
}

testSearch().catch(console.error);










// use this for unit test:  14-10-0081-01












