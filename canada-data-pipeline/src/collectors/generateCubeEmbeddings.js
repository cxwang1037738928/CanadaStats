// generateCubeEmbeddings.js
import fs from 'fs/promises';
import { pipeline } from '@xenova/transformers';

async function generateEmbeddings() {
  console.log('Loading model...');
  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  
  // Load your cube index
  const cubesRaw = await fs.readFile('./provincialCubeIndex.json', 'utf8');
  const cubes = JSON.parse(cubesRaw);
  
  console.log(`Generating embeddings for ${cubes.length} cubes...`);
  
  // Generate embeddings for each cube title
  const cubesWithEmbeddings = [];
  
  for (let i = 0; i < cubes.length; i++) {
    const cube = cubes[i];
    // Create searchable text from title and metadata
    const searchText = `${cube.title} ${cube.startDate || ''} ${cube.endDate || ''}`;
    
    const embedding = await extractor(searchText, { 
      pooling: 'mean', 
      normalize: true 
    });
    
    // Round each value to 3 decimal places for faster calculations
    const roundedEmbedding = Array.from(embedding.data).map(v => Math.round(v * 1000) / 1000);
    
    cubesWithEmbeddings.push({
      cubeId: cube.cubeId,
      title: cube.title,
      startDate: cube.startDate,
      endDate: cube.endDate,
      frequencyCode: cube.frequencyCode,
      geographyCount: cube.geographyCount,
      embedding: roundedEmbedding  // Now with 3 decimal precision
    });
    
    if ((i + 1) % 20 === 0) {
      console.log(`Processed ${i + 1}/${cubes.length} cubes`);
    }
  }
  
  // Save to ./cubesWithEmbeddings.json (not in public folder)
  await fs.writeFile(
    './cubesWithEmbeddings.json',
    JSON.stringify(cubesWithEmbeddings, null, 2)
  );
  
  // Calculate file size for reference
  const stats = await fs.stat('./cubesWithEmbeddings.json');
  const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);
  
  console.log(`✓ Saved cubesWithEmbeddings.json (${fileSizeMB} MB) with 3-decimal precision embeddings`);
  console.log(`  Each embedding has ${cubesWithEmbeddings[0]?.embedding.length || 0} dimensions`);
}

generateEmbeddings().catch(console.error);