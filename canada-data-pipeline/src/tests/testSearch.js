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
  
  // Find Geography dimension
  const geoDimension = metadata.dimension.find(dim => 
    dim.dimensionNameEn === "Geography" || dim.dimensionNameEn?.includes("Geography")
  );
  
  if (!geoDimension) {
    return { success: false, error: 'No geography dimension found' };
  }
  
  // Extract unit of measurement from dimensions
  let unitOfMeasurement = null;
  let scalarInfo = null;
  
  // Look for UOM dimension or scalar factor
  const uomDimension = metadata.dimension.find(dim => 
    dim.hasUOM === true || dim.dimensionNameEn?.toLowerCase().includes('unit')
  );
  
  if (uomDimension && uomDimension.member && uomDimension.member.length > 0) {
    const uomMember = uomDimension.member.find(m => m.memberUomCode) || uomDimension.member[0];
    if (uomMember && uomMember.memberNameEn) {
      unitOfMeasurement = uomMember.memberNameEn;
    }
  }
  
  // Get scalar factor info (thousands, millions, etc.)
  const scalarCodeSet = await fetchCodeSet('scalar');
  if (scalarCodeSet && metadata.scalarFactorCode) {
    scalarInfo = scalarCodeSet.find(s => s.scalarFactorCode === metadata.scalarFactorCode);
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
  
  // Collect data points
  const dataPoints = [];
  
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
            year: year,
            scalarFactor: point.scalarFactorCode
          });
        }
        return;
      }
      
      const dim = otherDimensions[dimIndex];
      const dimName = dim.dimensionNameEn;
      
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
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  // Build dimensions structure
  const dimensions = [
    { name: "Province", values: provinces.map(p => p.name) },
    ...otherDimensions.map(dim => ({
      name: dim.dimensionNameEn,
      values: dim.member?.filter(m => m.memberId !== 0).map(m => m.memberNameEn) || []
    }))
  ];
  
  const multiDimData = {
    cubeId: cubeId,
    title: cubeTitle,
    dimensions: dimensions,
    data: dataPoints,
    unitOfMeasurement: unitOfMeasurement,
    scalarInfo: scalarInfo,
    metadata: {
      startDate: metadata.cubeStartDate,
      endDate: metadata.cubeEndDate,
      frequencyCode: metadata.frequencyCode,
      footnotes: metadata.footnote?.slice(0, 3).map(f => f.footnotesEn) || []
    }
  };
  
  return { success: true, data: multiDimData };
}

async function fetchCodeSet(codeSetName) {
  try {
    const url = "https://www150.statcan.gc.ca/t1/wds/rest/getCodeSets";
    const response = await axios.get(url);
    if (response.data && response.data.status === "SUCCESS" && response.data.object) {
      return response.data.object[codeSetName];
    }
    return null;
  } catch (error) {
    return null;
  }
}

function formatValueWithUnit(value, unit, scalar) {
  let formattedValue = value;
  
  // Apply scalar factor (thousands, millions, etc.)
  if (scalar) {
    const scalarFactor = scalar.scalarFactorDescEn?.toLowerCase() || '';
    if (scalarFactor.includes('thousands')) {
      formattedValue = value * 1000;
    } else if (scalarFactor.includes('millions')) {
      formattedValue = value * 1000000;
    } else if (scalarFactor.includes('billions')) {
      formattedValue = value * 1000000000;
    }
  }
  
  // Format with appropriate commas
  const formattedNumber = Math.round(formattedValue).toLocaleString();
  
  // Add unit
  if (unit) {
    const unitLower = unit.toLowerCase();
    if (unitLower.includes('dollar') || unitLower.includes('cad')) {
      return '$' + formattedNumber;
    } else if (unitLower.includes('percent')) {
      return formattedNumber + '%';
    } else if (unitLower.includes('thousand') || unitLower.includes('thousands')) {
      return formattedNumber + ' (thousands)';
    } else if (unitLower.includes('million') || unitLower.includes('millions')) {
      return formattedNumber + ' (millions)';
    } else {
      return formattedNumber + ' ' + unit;
    }
  }
  
  return formattedNumber;
}

function displayMultiDimData(multiDimData) {
  console.log(`\n  📊 Unit of Measurement: ${multiDimData.unitOfMeasurement || 'Not specified'}`);
  if (multiDimData.scalarInfo) {
    console.log(`  📐 Scalar Factor: ${multiDimData.scalarInfo.scalarFactorDescEn || 'None'}`);
  }
  
  console.log(`\n  📐 Dimensions in this cube:`);
  multiDimData.dimensions.forEach((dim, idx) => {
    console.log(`     [${idx}] ${dim.name}: ${dim.values.length} unique values`);
    if (dim.values.length <= 5) {
      console.log(`         ${dim.values.join(', ')}`);
    } else {
      console.log(`         ${dim.values.slice(0, 5).join(', ')}... (${dim.values.length} total)`);
    }
  });
  
  console.log(`\n  📊 All data points:`);
  console.log('  ' + '-'.repeat(80));
  
  multiDimData.data.forEach(point => {
    const dimStr = Object.entries(point.dimensions)
      .filter(([k]) => k !== 'Province')
      .map(([k, v]) => `${k}=${v}`)
      .join(', ');
    
    const formattedValue = formatValueWithUnit(
      point.value, 
      multiDimData.unitOfMeasurement,
      multiDimData.scalarInfo
    );
    
    console.log(`     ${point.dimensions.Province.padEnd(25)} ${formattedValue.padEnd(20)} (${point.year})${dimStr ? ` [${dimStr}]` : ''}`);
  });
  
  // Create list-of-lists representation with unit
  console.log(`\n  📋 List-of-lists representation (for easy processing):`);
  console.log(`     dataAsListOfLists = [`);
  
  const unitLabel = multiDimData.unitOfMeasurement || 'Value';
  const headers = ['Province', `${unitLabel}`, 'Year'];
  console.log(`       ${JSON.stringify(headers)},`);
  
  multiDimData.data.forEach((point, idx) => {
    const formattedValue = formatValueWithUnit(
      point.value, 
      multiDimData.unitOfMeasurement,
      multiDimData.scalarInfo
    );
    const row = [point.dimensions.Province, formattedValue, point.year];
    console.log(`       ${JSON.stringify(row)}${idx === multiDimData.data.length - 1 ? '' : ','}`);
  });
  
  console.log(`     ]`);
  
  if (multiDimData.metadata.footnotes.length > 0) {
    console.log(`\n  📝 Notes: ${multiDimData.metadata.footnotes[0].substring(0, 200)}...`);
  }
  
  // Print summary for mapping
  console.log(`\n  🗺️ Ready for Canada Map:`);
  console.log(`     Metric: ${multiDimData.title.split(',')[0]}`);
  console.log(`     Unit: ${multiDimData.unitOfMeasurement || 'Count'}${multiDimData.scalarInfo ? ` (${multiDimData.scalarInfo.scalarFactorDescEn})` : ''}`);
  console.log(`     Year: ${multiDimData.data[0]?.year || 'N/A'}`);
}

async function testSearch() {
  const embeddingsPath = path.join(process.cwd(), '../collectors/cubesWithEmbeddings.json');
  console.log(`Loading embeddings from: ${embeddingsPath}`);
  const embeddingsData = await fs.readFile(embeddingsPath, 'utf8');
  const cubes = JSON.parse(embeddingsData);
  
  const userQuery = "provincial contribution to national budget";
  
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
    const result = await fetchMultiDimensionalData(cube.cubeId, cube.title);
    
    if (result.success && result.data.data.length > 0) {
      console.log(`\n✅ Success! Retrieved ${result.data.data.length} data points`);
      displayMultiDimData(result.data);
      
      if (!bestMultiDimData && result.data.data.length >= 10) {
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
    console.log(`\n✅ Best cube for mapping: ${bestMultiDimData.cubeId}`);
    console.log(`   ${bestMultiDimData.title}`);
    console.log(`\n   Unit: ${bestMultiDimData.unitOfMeasurement || 'Count'} ${bestMultiDimData.scalarInfo ? `(${bestMultiDimData.scalarInfo.scalarFactorDescEn})` : ''}`);
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