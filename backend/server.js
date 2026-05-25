import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios';
import { pipeline } from '@xenova/transformers';

const allowedOrigins = [
  'http://localhost:5173',                
  'http://localhost:3000',                 
  'https://your-frontend.vercel.app'       
];

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app  = express();
const PORT = process.env.BACKEND_PORT || process.env.PORT || 9999;

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps, curl, or Postman)
    if (!origin) return callback(null, true);
    
    if (allowedOrigins.indexOf(origin) === -1) {
      const msg = 'The CORS policy for this site does not allow access from the specified Origin.';
      return callback(new Error(msg), false);
    }
    return callback(null, true);
  },
  credentials: true // Enable this if you pass cookies or authorization headers
}));
app.use(express.json());

let cachedCubes    = null;
let embeddingModel = null;

// ── Startup loaders ───────────────────────────────────────────────────────────
async function loadCubes() {
  if (cachedCubes) return cachedCubes;
  const p = path.join(__dirname, '../canada-data-pipeline/src/collectors/cubesWithEmbeddings.json');
  cachedCubes = JSON.parse(await fs.readFile(p, 'utf8'));
  console.log(`Loaded ${cachedCubes.length} cubes`);
  return cachedCubes;
}

async function getEmbeddingModel() {
  if (!embeddingModel) {
    console.log('Loading embedding model…');
    embeddingModel = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log('Model ready');
  }
  return embeddingModel;
}

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
const AGGREGATE_KW = ['total', 'both sexes', 'both genders', 'all ages', 'all ', 'aggregate'];

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
    const pt = r.data?.[0]?.object?.vectorDataPoint?.[0];
    if (!pt || pt.value == null) return null;
    const decimals = pt.decimals ?? 0;
    const value = decimals > 0
      ? Math.round(pt.value * 10 ** decimals) / 10 ** decimals
      : Number(pt.value);
    return { value, year: pt.refPer?.split('-')[0] ?? 'N/A' };
  } catch { return null; }
}

// ── POST /api/search ──────────────────────────────────────────────────────────
// Returns cube metadata + dimension structure. No data fetched yet.
// Response: { cubeId, title, unit, tableUrl, dimensionMeta, geoDimIndex, provinces }
// dimensionMeta: [{ name, dimIndex, members: [{ name, memberId, isAggregate }] }]
// provinces: [{ name, memberId }]
app.post('/api/search', async (req, res) => {
  try {
    const { query, topK = 5 } = req.body;
    if (!query?.trim()) return res.status(400).json({ error: 'Query is required' });

    console.log(`\n${'─'.repeat(60)}\nSearching: "${query}"`);

    const cubes = await loadCubes();
    const model = await getEmbeddingModel();
    const emb   = await model(query, { pooling: 'mean', normalize: true });
    const qVec  = Array.from(emb.data);

    const ranked = cubes
      .map(c => ({ cubeId: c.cubeId, title: c.title, similarity: cosineSimilarity(qVec, c.embedding) }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);

    console.log('Top results:');
    ranked.forEach((r, i) =>
      console.log(`  ${i+1}. [${r.cubeId}] ${r.title.slice(0,60)}… (${(r.similarity*100).toFixed(1)}%)`)
    );

    // Find the first cube that has a Geography dimension with known provinces
    for (const candidate of ranked) {
      const metadata = await fetchMeta(candidate.cubeId);
      if (!metadata) continue;

      const geoDim = metadata.dimension.find(d =>
        d.dimensionNameEn === 'Geography' || d.dimensionNameEn?.includes('Geography')
      );
      if (!geoDim) continue;

      const provinces = geoDim.member
        .filter(m => PROVINCE_MAPPING[m.memberNameEn])
        .map(m => ({ name: PROVINCE_MAPPING[m.memberNameEn], memberId: m.memberId }));

      if (provinces.length < 8) continue; // not enough geographic coverage

      // Unit of measurement
      let unit = null;
      const uomDim = metadata.dimension.find(d => d.hasUOM === true);
      if (uomDim?.member?.length) {
        const m = uomDim.member.find(m => m.memberUomCode) ?? uomDim.member[0];
        unit = m?.memberNameEn ?? null;
      }

      // Build dimension metadata for every non-geo dimension
      const geoDimIndex = metadata.dimension.findIndex(d =>
        d.dimensionNameEn === 'Geography' || d.dimensionNameEn?.includes('Geography')
      );

      const dimensionMeta = metadata.dimension
        .map((dim, idx) => ({ dim, idx }))
        .filter(({ idx }) => idx !== geoDimIndex)
        .map(({ dim, idx }) => ({
          name:     dim.dimensionNameEn,
          dimIndex: idx,           // position in the 10-slot coordinate array
          members:  (dim.member ?? [])
            .filter(m => m.memberId && m.memberId !== 0)
            .map(m => ({
              name:        m.memberNameEn,
              memberId:    m.memberId,
              isAggregate: isAggregate(m.memberNameEn),
            })),
        }));

      // StatCan table URL (format: pid as 8-digit zero-padded, then -01)
      const pid     = String(candidate.cubeId).padStart(8, '0');

      // Clean the input PID (removes hyphens if present) and appends the "01" suffix
      const cleanPid = pid.replace(/-/g, '');
      const fullPid = cleanPid.endsWith('01') ? cleanPid : `${cleanPid}01`;

      // The correct StatCan interactive table URL format
      const tableUrl = `https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=${fullPid}`;

      console.log(`✅ Using cube ${candidate.cubeId}: ${candidate.title.slice(0, 60)}`);

      return res.json({
        success: true,
        cubeId:       candidate.cubeId,
        title:        candidate.title,
        unit,
        tableUrl,
        geoDimIndex,
        provinces,       // [{ name, memberId }]
        dimensionMeta,   // [{ name, dimIndex, members[] }]
      });
    }

    return res.status(404).json({ error: 'No suitable cube found with provincial data' });

  } catch (err) {
    console.error('Search error:', err);
    res.status(500).json({ error: 'Internal server error', message: err.message });
  }
});

// ── POST /api/data ────────────────────────────────────────────────────────────
// Called when the user picks dimension members. Fetches one coordinate per province.
// Body: { cubeId, geoDimIndex, selections: { dimIndex: memberId, … } }
// selections maps each non-geo dimension's index to the chosen memberId.
// Response: { provinces: [{ province, value, year }], year }
app.post('/api/data', async (req, res) => {
  try {
    const { cubeId, geoDimIndex, provinces, selections } = req.body;
    // provinces: [{ name, memberId }]
    // selections: { "2": 3, "3": 7 }  — dimIndex → memberId

    if (!cubeId || !provinces?.length || !selections) {
      return res.status(400).json({ error: 'cubeId, provinces, and selections are required' });
    }

    console.log(`\nFetching data for cube ${cubeId}`);
    console.log(`Selections:`, selections);

    const results = [];

    for (const province of provinces) {
      // Build 10-slot coordinate string
      const coord = Array(10).fill('0');

      // Slot for geography (dimIndex is 0-based; coord slots are also 0-based)
      coord[geoDimIndex] = province.memberId.toString();

      // Fill in user-selected members
      for (const [dimIdx, memberId] of Object.entries(selections)) {
        coord[parseInt(dimIdx)] = memberId.toString();
      }

      const coordinateStr = coord.join('.');
      console.log(`  ${province.name}: ${coordinateStr}`);

      const result = await fetchCoordinate(cubeId, coordinateStr);
      if (result) {
        results.push({ province: province.name, value: result.value, year: result.year });
      }

      await new Promise(r => setTimeout(r, 60)); // polite delay
    }

    if (!results.length) {
      return res.status(404).json({ error: 'No data found for this combination' });
    }

    const years = results.map(r => r.year).filter(y => y !== 'N/A');
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

app.get('/api/health', (_req, res) =>
  res.json({ status: 'healthy', cubesLoaded: !!cachedCubes })
);

app.listen(PORT, () => console.log(`\n🚀 http://localhost:${PORT}`));
loadCubes().catch(console.error);
