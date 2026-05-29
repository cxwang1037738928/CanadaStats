import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios';
import { pipeline } from '@xenova/transformers';

const allowedOrigins = [
   "https://canadamapped.ca",
   "https://www.canadamapped.ca",
  'http://localhost:5173',                
  'http://localhost:3000',                 
  process.env.FRONTEND_URL // Pulls live Vercel URL dynamically from AWS Env Variables
].filter(Boolean); // Cleans out undefined/empty values

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app  = express();
const PORT = process.env.BACKEND_PORT || process.env.PORT || 9999;

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps, curl, or Postman)
    if (!origin) return callback(null, true);

    const isAllowed = allowedOrigins.includes(origin);
    const isVercelPreview = origin.endsWith('.vercel.app'); // only allow vercel

    if (isAllowed || isVercelPreview) {
      return callback(null, true);
    }
    
    // Safe rejection: tell the browser "No" without crashing the server with a 500
    return callback(null, false); 
  },
  credentials: true 
}));


app.use(express.json());
// In-memory cache for cube metadata and embeddings to avoid file reads on every request
let cachedCubes    = null;
let embeddingModel = null;

// ── Startup loaders ───────────────────────────────────────────────────────────
// loads the cube metadata from the pre-generated file with embeddings
async function loadCubes() {
  if (cachedCubes) return cachedCubes; // return cached version if already loaded, saves file read time on subsequent requests
  
  // two paths in case of different launch contexts(in case of hosting only backend)  
  // Path option A: If you launched Node from the project ROOT directory
  const rootWorkspacePath = path.join(process.cwd(), 'canada-data-pipeline', 'src', 'collectors', 'cubesWithEmbeddings.json');
  
  // Path option B: If you launched Node from INSIDE the /backend folder
  const internalBackendPath = path.join(__dirname, '../canada-data-pipeline/src/collectors/cubesWithEmbeddings.json');

  let resolvedPath;
  try {
    // Check if the root workspace layout can see the file
    await fs.access(rootWorkspacePath);
    resolvedPath = rootWorkspacePath;
  } catch {
    // If option A fails, default to relative directory hopping
    resolvedPath = internalBackendPath;
  }

  console.log(`📂 Database file resolved at: ${resolvedPath}`);
  
  try {
    const rawData = await fs.readFile(resolvedPath, 'utf8');
    cachedCubes = JSON.parse(rawData);
    console.log(`Loaded ${cachedCubes.length} cubes successfully.`);
    return cachedCubes;
  } catch (readError) {
    console.error(`❌ Critical error reading the file at ${resolvedPath}:`, readError.message);
    throw readError;
  }
}
// load the embedding once once at startup anc cache it
async function getEmbeddingModel() {
  if (!embeddingModel) {
    console.log('Loading embedding model…');
    embeddingModel = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log('Model ready');
  }
  return embeddingModel;
}
// normalized dot product similarity since all vectors are normalized to length 1, so we can skip the denominator for faster calculations
function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// ── StatCan constants ─────────────────────────────────────────────────────────
const STATCAN = 'https://www150.statcan.gc.ca/t1/wds/rest';

const PROVINCE_MAPPING = {
  'Newfoundland and Labrador': 'Newfoundland and Labrador',
  'Prince Edward Island':      'Prince Edward Island',
  'Nova Scotia':               'Nova Scotia',
  'New Brunswick':             'New Brunswick',
  'Quebec':                    'Quebec',
  'Ontario':                   'Ontario',
  'Manitoba':                  'Manitoba',
  'Saskatchewan':              'Saskatchewan',
  'Alberta':                   'Alberta',
  'British Columbia':          'British Columbia',
};

// Keywords that identify an "aggregate / total" member so we can default to it
// not perfect since for number of students for example, this would work well
// but if its unemployment rate, then it just adds up the rate for both genders instead of averaging it.
const AGGREGATE_KW = ['total', 'both sexes', 'both genders', 'all ages', 'all ', 'aggregate'];

// Simple heuristic to identify if a member is an aggregate
function isAggregate(name) {
  const l = (name ?? '').toLowerCase();
  return AGGREGATE_KW.some(k => l.includes(k));
}

// ── Fetch helpers ─────────────────────────────────────────────────────────────
async function fetchMeta(cubeId) {
  const r = await axios.post(
    `${STATCAN}/getCubeMetadata`,
    [{ productId: parseInt(cubeId) }],
    { headers: { 'Content-Type': 'application/json' }, timeout: 10000 }
  );
  return r.data?.[0]?.object ?? null;
}

// Fetch a single coordinate for a single province — returns { value, year } or null
async function fetchCoordinate(cubeId, coordinate) {
  try {
    const r = await axios.post(
      `${STATCAN}/getDataFromCubePidCoordAndLatestNPeriods`,
      [{ productId: parseInt(cubeId), coordinate, latestN: 1 }],
      { headers: { 'Content-Type': 'application/json' }, timeout: 10000 }
    );
    // single point response, object contains the data, vectorDataPoint is an array of datapoints
    // only the first (the latest) datapoint is returned
    const pt = r.data?.[0]?.object?.vectorDataPoint?.[0];

    if (!pt || pt.value == null) return null;
    const decimals = pt.decimals ?? 0;
    const value = decimals > 0
      ? Math.round(pt.value * 10 ** decimals) / 10 ** decimals
      : Number(pt.value);
    // returns year, gets rid of the '-' from the period string
    return { value, year: pt.refPer?.split('-')[0] ?? 'N/A' };
  } catch { return null; }
}

// ── POST /api/search ──────────────────────────────────────────────────────────
// compares query embedding with the store cube embeddings and finds the most semanticaly similar cubes
// then validates that the cube contains provincial data and extract the meta data about
// dimensions and provinces
app.post('/api/search', async (req, res) => {
  try {
    const { query, topK = 5 } = req.body;
    if (!query?.trim()) return res.status(400).json({ error: 'Query is required' });

    console.log(`\n${'─'.repeat(60)}\nSearching: "${query}"`);
    
    const cubes = await loadCubes();
    const model = await getEmbeddingModel();
    const emb   = await model(query, { pooling: 'mean', normalize: true });
    const qVec  = Array.from(emb.data);
    
    // cube structure: { cubeId, title, embedding}
    // performs a cosine similary check between the query embedding and each
    // cube embedding, then sorts by similarity and takes the top K results
    const ranked = cubes
      .map(c => ({ cubeId: c.cubeId, title: c.title, similarity: cosineSimilarity(qVec, c.embedding) }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
    
    console.log('Top results:');
    // logs the top K results with their similarity scores
    ranked.forEach((r, i) =>
      console.log(`  ${i+1}. [${r.cubeId}] ${r.title.slice(0,60)}… (${(r.similarity*100).toFixed(1)}%)`)
    );
    // fetches the metadata for each of the top K cubes
    for (const candidate of ranked) {
      const metadata = await fetchMeta(candidate.cubeId);
      if (!metadata) continue;
      // locates the geography dimension by looking for keywords
      const geoDim = metadata.dimension.find(d =>
        d.dimensionNameEn === 'Geography' || d.dimensionNameEn?.includes('Geography')
      );
      // does not consider cubes that do not have a geography dimension, since we need provincial data for the map
      if (!geoDim) continue;
      // removes members that are not provinces based on PROVINCE_MAPPING
      const provinces = geoDim.member
        .filter(m => PROVINCE_MAPPING[m.memberNameEn])
        // transforms the member list into a list of objects with province name and memberId
        .map(m => ({ name: PROVINCE_MAPPING[m.memberNameEn], memberId: m.memberId }));

      // if there are less than 8 provinces, then the cube is skipped
      if (provinces.length < 8) continue; 

      let unit = null;

      const uomDim = metadata.dimension.find(d => d.hasUOM === true);
      // attempts to find the UoM, #TODO: this always fails so it needs another look in the future
      if (uomDim?.member?.length) {
        const m = uomDim.member.find(m => m.memberUomCode) ?? uomDim.member[0];
        unit = m?.memberNameEn ?? null;
      }
      
      // finds geography dimension again, guaranteed to succeed since its already checked above
      const geoDimIndex = metadata.dimension.findIndex(d =>
        d.dimensionNameEn === 'Geography' || d.dimensionNameEn?.includes('Geography')
      );
      
      const dimensionMeta = metadata.dimension
        // creates a new array where each dimension object is paired with its index in the array
        .map((dim, idx) => ({ dim, idx }))
        // removes geography dimension 
        .filter(({ idx }) => idx !== geoDimIndex)
        // transform each remaining dimension into:
        /** {
              name: "Time",
              dimIndex: 1,
              members: [
                { name: "2020", memberId: 101, isAggregate: false },
                { name: "2021", memberId: 102, isAggregate: false }
              ]
            },
            {
              name: "Sex",
              dimIndex: 2,
              members: [
                { name: "Male", memberId: 1, isAggregate: false },
                { name: "Female", memberId: 2, isAggregate: false }
              ]
            }
          ];
         * 
         */
        .map(({ dim, idx }) => ({
          name:     dim.dimensionNameEn,
          dimIndex: idx,           
          members:  (dim.member ?? []) // use an empty array if there are no members
            .filter(m => m.memberId && m.memberId !== 0)
            .map(m => ({
              name:        m.memberNameEn,
              memberId:    m.memberId,
              isAggregate: isAggregate(m.memberNameEn),
            })),
        }));
      // padds it with leading zeros until it is 8 units long
      const pid     = String(candidate.cubeId).padStart(8, '0');
      // removes any hyphens from the cube ID
      const cleanPid = pid.replace(/-/g, '');

      // statcan table urls require a suffix like 01, if the cube already ends with 01 then don't do anything
      const fullPid = cleanPid.endsWith('01') ? cleanPid : `${cleanPid}01`;
      // full table URL
      const tableUrl = `https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=${fullPid}`;

      console.log(`✅ Using cube ${candidate.cubeId}: ${candidate.title.slice(0, 60)}`);

      return res.json({
        success: true,
        cubeId:       candidate.cubeId,
        title:        candidate.title,
        unit,
        tableUrl,
        geoDimIndex,
        provinces,       
        dimensionMeta,   
      });
    }

    return res.status(404).json({ error: 'No suitable cube found with provincial data' });

  } catch (err) {
    console.error('Search error:', err);
    res.status(500).json({ error: 'Internal server error', message: err.message });
  }
});

// ── POST /api/data ────────────────────────────────────────────────────────────
// Builds StatCan coordinates per province and then fetch the data for each province
// based on the selected cube and dimensions
app.post('/api/data', async (req, res) => {
  try {
    const { cubeId, geoDimIndex, provinces, selections } = req.body;

    if (!cubeId || !provinces?.length || !selections) {
      return res.status(400).json({ error: 'cubeId, provinces, and selections are required' });
    }

    console.log(`\nFetching data for cube ${cubeId}`);
    console.log(`Selections:`, selections);

    const results = [];
    
    for (const province of provinces) {
      // initializes [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      const coord = Array(10).fill('0');
      // sets the geography dimension to the current province memberId
      coord[geoDimIndex] = province.memberId.toString();

      // iterates through the other dimensions the user selected and sets the coordinates
      for (const [dimIdx, memberId] of Object.entries(selections)) {
        coord[parseInt(dimIdx)] = memberId.toString();
      }
      // converts array into single string with each coordinate seperated by a dot
      // e.g "0.3.35.0.7.0.0.0.0.0"
      const coordinateStr = coord.join('.');
      console.log(`  ${province.name}: ${coordinateStr}`);
      // fetch the data for the current province and coordinates
      const result = await fetchCoordinate(cubeId, coordinateStr);
      if (result) {
        results.push({ province: province.name, value: result.value, year: result.year });
      }
      // prevents too many API requests to StatCan in a short period of time
      // since StatCan limits requests from servers to 50 per second
      await new Promise(r => setTimeout(r, 60)); 
    }

    if (!results.length) {
      return res.status(404).json({ error: 'No data found for this combination' });
    }
    // takes only the year value from results
    const years = results.map(r => r.year).filter(y => y !== 'N/A');
    // sorts the year by frequency and takes the most common year among the results
    const year  = years.sort((a, b) =>
      years.filter(v => v === b).length - years.filter(v => v === a).length
    )[0] ?? 'N/A';

    console.log(`  → ${results.length} provinces returned`);
    return res.json({ success: true, provinces: results, year });

  } catch (err) {
    console.error('Data fetch error:', err);
    res.status(500).json({ error: 'Internal server error', message: err.message });
  }
});
// check if the server is running and if the cubes are loaded in memory
app.get('/api/health', (_req, res) =>
  res.json({ status: 'healthy', cubesLoaded: !!cachedCubes })
);

app.listen(PORT, () => console.log(`\n🚀 Server running on port ${PORT}`));
loadCubes().catch(console.error);