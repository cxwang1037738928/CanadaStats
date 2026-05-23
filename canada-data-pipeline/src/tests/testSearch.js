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

async function fetchMultiDimensionalData(cubeId, cubeTitle) {
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
  
  // Find Geography dimension (always dimension 0 in our coordinates)
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
  
  // Identify other dimensions (excluding Geography)
  const otherDimensions = metadata.dimension.filter(dim => 
    dim.dimensionNameEn !== "Geography" && !dim.dimensionNameEn?.includes("Geography")
  );
  
  // Build dimension structure
  const dimensions = [
    { name: "Province", values: provinces.map(p => p.name) },
    ...otherDimensions.map(dim => ({
      name: dim.dimensionNameEn,
      values: dim.member?.filter(m => m.memberId !== 0).map(m => m.memberNameEn) || []
    }))
  ];
  
  // Collect data for all combinations
  const dataPoints = [];
  
  // For each province
  for (const province of provinces) {
    // Build base coordinate with province
    const baseCoordinate = Array(10).fill('0');
    baseCoordinate[0] = province.id.toString();
    
    // Get values for each combination of other dimensions
    // To avoid too many API calls, we'll use a recursive approach
    async function exploreDimension(dimIndex, currentCoordinate, currentSelections) {
      if (dimIndex >= otherDimensions.length) {
        // Fetch data for this coordinate
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
            dimensions: {
              Province: province.name,
              ...currentSelections
            },
            value: value,
            year: year
          });
        }
        return;
      }
      
      const dim = otherDimensions[dimIndex];
      const dimName = dim.dimensionNameEn;
      
      // Find "Total" or "All" or "Both sexes" as default
      const totalMember = dim.member?.find(m => 
        m.memberNameEn?.toLowerCase().includes('total') || 
        m.memberNameEn?.toLowerCase().includes('all') ||
        m.memberNameEn?.toLowerCase().includes('both sexes') ||
        m.memberNameEn?.toLowerCase().includes('both genders') ||
        m.memberNameEn === 'Both sexes' ||
        m.memberNameEn === 'Total'
      );
      
      if (totalMember) {
        currentCoordinate[dimIndex + 1] = totalMember.memberId.toString();
        await exploreDimension(dimIndex + 1, currentCoordinate, {
          ...currentSelections,
          [dimName]: totalMember.memberNameEn
        });
      } else if (dim.member && dim.member.length > 0) {
        // Take first non-zero member
        const firstMember = dim.member.find(m => m.memberId !== 0);
        if (firstMember) {
          currentCoordinate[dimIndex + 1] = firstMember.memberId.toString();
          await exploreDimension(dimIndex + 1, currentCoordinate, {
            ...currentSelections,
            [dimName]: firstMember.memberNameEn
          });
        }
      }
    }
    
    await exploreDimension(0, [...baseCoordinate], {});
    await new Promise(resolve => setTimeout(resolve, 100)); // Rate limiting
  }
  
  // Organize data into multi-dimensional list structure
  const multiDimData = {
    cubeId: cubeId,
    title: cubeTitle,
    dimensions: dimensions,
    data: dataPoints,
    valueDescription: getValueDescription(cubeTitle),
    metadata: {
      startDate: metadata.cubeStartDate,
      endDate: metadata.cubeEndDate,
      frequencyCode: metadata.frequencyCode,
      footnotes: metadata.footnote?.slice(0, 3).map(f => f.footnotesEn) || []
    }
  };
  
  return { success: true, data: multiDimData };
}

function getValueDescription(cubeTitle) {
  const title = cubeTitle.toLowerCase();
  
  if (title.includes('median') && title.includes('income')) {
    return 'Median income (CAD $)';
  } else if (title.includes('average') && title.includes('income')) {
    return 'Average income (CAD $)';
  } else if (title.includes('gini')) {
    return 'Gini coefficient (0=perfect equality, 1=perfect inequality)';
  } else if (title.includes('population')) {
    return 'Population count';
  } else if (title.includes('employment')) {
    return 'Employment count';
  } else if (title.includes('unemployment')) {
    return 'Unemployment rate (%)';
  } else if (title.includes('inequality')) {
    return 'Gini coefficient (income inequality measure)';
  }
  
  return 'Statistical value';
}

function formatValueWithContext(value, description) {
  if (description.includes('Gini')) {
    return value.toFixed(3);
  } else if (description.includes('rate (%)')) {
    return value.toFixed(1) + '%';
  } else if (description.includes('income') && description.includes('CAD')) {
    return '$' + value.toLocaleString();
  } else if (description.includes('income')) {
    return '$' + value.toLocaleString();
  } else {
    return value.toLocaleString();
  }
}

function displayMultiDimData(multiDimData) {
  console.log(`\n  📊 Value represents: ${multiDimData.valueDescription}`);
  console.log(`\n  📐 Dimensions in this cube:`);
  multiDimData.dimensions.forEach((dim, idx) => {
    console.log(`     [${idx}] ${dim.name}: ${dim.values.length} unique values`);
    if (dim.values.length <= 5) {
      console.log(`         ${dim.values.join(', ')}`);
    } else {
      console.log(`         ${dim.values.slice(0, 5).join(', ')}... (${dim.values.length} total)`);
    }
  });
  
  console.log(`\n  📋 Data structure:`);
  console.log(`     The data is a list of objects, each with:`);
  console.log(`     - dimensions: { Province: "...", [other dimensions]: "..." }`);
  console.log(`     - value: number`);
  console.log(`     - year: string`);
  
  console.log(`\n  📊 Sample data points (first 10):`);
  console.log('  ' + '-'.repeat(80));
  
  multiDimData.data.slice(0, 10).forEach(point => {
    const dimStr = Object.entries(point.dimensions).map(([k, v]) => `${k}=${v}`).join(', ');
    const formattedValue = formatValueWithContext(point.value, multiDimData.valueDescription);
    console.log(`     { ${dimStr}, value: ${formattedValue}, year: ${point.year} }`);
  });
  
  if (multiDimData.data.length > 10) {
    console.log(`     ... and ${multiDimData.data.length - 10} more data points`);
  }
  
  // Create list-of-lists representation for easy processing
  console.log(`\n  📋 List-of-lists representation (for easy processing):`);
  console.log(`     dataAsListOfLists = [`);
  
  // Headers row
  const headers = ['Province', ...multiDimData.dimensions.slice(1).map(d => d.name), 'Value', 'Year'];
  console.log(`       ${JSON.stringify(headers)},`);
  
  // Data rows
  const sampleRows = multiDimData.data.slice(0, 5);
  sampleRows.forEach(point => {
    const row = [
      point.dimensions.Province,
      ...multiDimData.dimensions.slice(1).map(dim => point.dimensions[dim.name] || 'Total'),
      formatValueWithContext(point.value, multiDimData.valueDescription),
      point.year
    ];
    console.log(`       ${JSON.stringify(row)}${point === sampleRows[sampleRows.length - 1] ? '' : ','}`);
  });
  
  if (multiDimData.data.length > 5) {
    console.log(`       ... and ${multiDimData.data.length - 5} more rows`);
  }
  console.log(`     ]`);
  
  if (multiDimData.metadata.footnotes.length > 0) {
    console.log(`\n  📝 Notes: ${multiDimData.metadata.footnotes[0].substring(0, 200)}...`);
  }
}

async function testSearch() {
  // Load cubesWithEmbeddings.json from ../collectors/
  const embeddingsPath = path.join(process.cwd(), '../collectors/cubesWithEmbeddings.json');
  console.log(`Loading embeddings from: ${embeddingsPath}`);
  const embeddingsData = await fs.readFile(embeddingsPath, 'utf8');
  const cubes = JSON.parse(embeddingsData);
  
  const userQuery = "percentage increase in population";
  
  console.log(`\nQuery: "${userQuery}"`);
  console.log(`Total cubes: ${cubes.length}\n`);
  
  console.log('Finding best cube for multi-dimensional data...');
  const start = Date.now();
  const results = await semanticSearch(userQuery, cubes, 10);
  
  let bestMultiDimData = null;
  
  for (let i = 0; i < Math.min(results.length, 3); i++) {
    const cube = results[i];
    console.log(`\n${'='.repeat(80)}`);
    console.log(`Testing cube ${i + 1}: ${cube.cubeId}`);
    console.log(`Title: ${cube.title.substring(0, 80)}...`);
    console.log(`Similarity: ${(cube.similarity * 100).toFixed(1)}%`);
    
    console.log(`\nFetching multi-dimensional data...`);
    const result = await fetchMultiDimensionalData(cube.cubeId, cube.title);
    
    if (result.success && result.data.data.length > 0) {
      console.log(`\n✅ Success! Retrieved ${result.data.data.length} data points`);
      displayMultiDimData(result.data);
      
      if (!bestMultiDimData && result.data.data.length >= 10) {
        bestMultiDimData = result.data;
      }
      break; // Stop after first successful cube
    } else {
      console.log(`   ⚠ ${result.error || 'No data retrieved'}`);
    }
  }
  
  console.log(`\n${'='.repeat(80)}`);
  console.log(`Search completed in ${Date.now() - start}ms`);
  
  if (bestMultiDimData) {
    console.log(`\n✅ Best cube for mapping: ${bestMultiDimData.cubeId}`);
    console.log(`   ${bestMultiDimData.title}`);
    console.log(`\n   This data structure includes:`);
    bestMultiDimData.dimensions.forEach(dim => {
      console.log(`   - ${dim.name} (${dim.values.length} values)`);
    });
    console.log(`\n   To use this data for a Canada map:`);
    console.log(`   1. Filter by the dimensions you want (e.g., Gender="Both sexes")`);
    console.log(`   2. Extract Province and Value`);
    console.log(`   3. Map value to province color`);
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