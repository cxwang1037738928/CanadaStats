import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios'; // auto built in json parsing and sets timeout
import { pipeline } from '@xenova/transformers';
import {
  getMlpModel,
  rerankByCategory,
  enrichWithMetadata,
  isCurrentCube,
  CLASSIFIER_EMBEDDING_MODEL
} from './Search_Utils.js';

// const EMBEDDINGS_FILENAME = 'cubesWithEmbeddings.all-mpnet-base-v2.json';
const EMBEDDINGS_FILENAME = 'cubesWithEmbeddings.all-MiniLM-L12-v2.json'; // matches CLASSIFIER_EMBEDDING_MODEL

const allowedOrigins = [
   "https://canadamapped.ca",
   "https://www.canadamapped.ca",
  'http://localhost:5173',        // for local development                      
  process.env.FRONTEND_URL // Pulls live Vercel URL dynamically from AWS Env Variables
].filter(Boolean); // Cleans out undefined/empty values

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app  = express();
// Backend port is used only for local development
const PORT = process.env.BACKEND_PORT || process.env.PORT || 5000; // Don't remove process.env.PORT since render relies on it.


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
let mlpWarmedUp    = false; // tracks whether the MLP query classifier (Search_Utils.js) has finished its startup warm-up

// In-flight promises for the two lazy singletons below. Without these, requests
// that arrive while a cold start is still loading each kick off their OWN copy
// of the work — several concurrent 20 MB JSON parses and duplicate embedding
// model loads. On a small instance that memory spike can OOM the process, which
// looks like a random server error to the user. Caching the promise (not just
// the result) means concurrent callers all await the same single load.
let cubesLoadPromise = null;
let embeddingLoadPromise = null;

// ── Startup loaders ───────────────────────────────────────────────────────────
// loads the cube metadata as json objects from the pre-generated file with embeddings
// NOTE: this must stay in sync with CLASSIFIER_EMBEDDING_MODEL (Search_Utils.js) —
// the MLP query classifier was trained on all-MiniLM-L12-v2 vectors, so the
// search embeddings and the search-time embedder both need to use that same
// model or category re-ranking (rerankByCategory) will be comparing vectors
// from different embedding spaces.
async function loadCubes() {
  if (cachedCubes) return cachedCubes; // return cached version if already loaded, saves file read time on subsequent requests
  // A load is already running — join it instead of starting a second one.
  if (cubesLoadPromise) return cubesLoadPromise;

  cubesLoadPromise = loadCubesUncached()
    .finally(() => { cubesLoadPromise = null; }); // clear so a failed load can be retried
  return cubesLoadPromise;
}

async function loadCubesUncached() {
  // two paths in case of different launch contexts(in case of hosting only backend)
  // Path option A: If you launched Node from the project ROOT directory
  const rootWorkspacePath = path.join(process.cwd(), 'canada-data-pipeline', 'src', 'collectors', EMBEDDINGS_FILENAME);
  
  // Path option B: If you launched Node from INSIDE the /backend folder
  const internalBackendPath = path.join(__dirname, '../canada-data-pipeline/src/collectors', EMBEDDINGS_FILENAME);

  let resolvedPath;
  try {
    // Check if the root workspace layout can see the file
    await fs.access(rootWorkspacePath);
    resolvedPath = rootWorkspacePath;
  } catch {
    // If option A fails, default to relative directory hopping
    resolvedPath = internalBackendPath;
  }

  console.log(` Database file resolved at: ${resolvedPath}`);
  
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
// Uses CLASSIFIER_EMBEDDING_MODEL (all-MiniLM-L12-v2) rather than a separately
// hardcoded model string, so the search embedder can never drift out of sync
// with the model the MLP query classifier (Search_Utils.js) was trained on.
// Outputs tensor with the properties below:
// Tensor {
//   dims: [ 2, 384 ],
//   type: 'float32',
//   data: Float32Array(384) [ 0.04592696577310562, 0.07328180968761444, ... ],
//   size: 384
// }
async function getEmbeddingModel() {
  if (embeddingModel) return embeddingModel;
  // Same in-flight de-duplication as loadCubes() — concurrent cold-start
  // requests must not each load their own copy of the model.
  if (embeddingLoadPromise) return embeddingLoadPromise;

  embeddingLoadPromise = (async () => {
    console.log(`Loading embedding model (${CLASSIFIER_EMBEDDING_MODEL})…`);
    embeddingModel = await pipeline('feature-extraction', CLASSIFIER_EMBEDDING_MODEL);
    console.log('Model ready');
    return embeddingModel;
  })().finally(() => { embeddingLoadPromise = null; }); // clear so a failed load can be retried

  return embeddingLoadPromise;
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

const AGGREGATE_KW = ['total', 'both sexes', 'both genders', 'all ages', 'all ', 'aggregate'];

// Simple heuristic to identify if a member is an aggregate
function isAggregate(name) {
  const l = (name ?? '').toLowerCase();
  return AGGREGATE_KW.some(k => l.includes(k));
}
// Fetches the cube metadata for a given cubeId, returns an object with the cube metadata
/* Cube metadata example:
[
{
"status": "SUCCESS",
"object": {
"responseStatusCode": 0,
"productId": "35100003",
"cansimId": "251-0008",
"cubeTitleEn": "Average counts of young persons in provincial and territorial correctional services",
"cubeTitleFr": "Comptes moyens des adolescents dans les services correctionnels provinciaux et territoriaux",
"cubeStartDate": "1997-01-01",
"cubeEndDate": "2015-01-01",
"frequencyCode": 12,
"nbSeriesCube": 171,
"nbDatapointsCube": 3129,
"releaseTime": "2015-05-09T08:30",
"archiveStatusCode": "2",
"archiveStatusEn": "CURRENT - a cube available to the public and that is current",
"archiveStatusFr": "ACTIF - un cube qui est disponible au public et qui est toujours mise a jour",
"subjectCode": [
"350102",
"4211"
],
"surveyCode": [
"3313"
],
"dimension": [
{
"dimensionPositionId": 1,
"dimensionNameEn": "Geography",
"dimensionNameFr": "Géographie",
"hasUom": false,
"member": [
{
"memberId": 1,
"parentMemberId": 15,
"memberNameEn": "Newfoundland and Labrador",
"memberNameFr": "Terre-Neuve-et-Labrador",
"classificationCode": "10",
"classificationTypeCode": "1",
"geoLevel": 2,
"vintage": 2011,
"terminated": 0,
"memberUomCode": null
},
… repeating objects
"footnote":[{"footnoteId":1,"footnotesEn":"Corrections Key Indicator Report for Youth, Canadian Centre for Justice and Community Safety Statistics (CCJCSS), Statistics Canada. Fiscal year (April 1 through March 31). Due to rounding,
… repeating objects
"link":{"footnoteId":22,"dimensionPositionId":2,"memberId":12}}],"correctionFootnote":[],"geoAttribute":[],"correction":[],"issueDate":"2021-04-13"}}]
*/
// ── Fetch helpers ─────────────────────────────────────────────────────────────
// StatCan drops connections (ECONNRESET) and times out intermittently, so a
// single attempt is not reliable. But retrying is only worth it for errors that
// might succeed next time — a 406 for a malformed product ID never will, and
// retrying it just burns the caller's latency budget.
const REQUEST_TIMEOUT_MS = 5000;  // StatCan normally answers well under 1s; a longer
                                  // wait almost always means a dead connection.
const MAX_ATTEMPTS       = 3;
const RETRY_BASE_MS      = 120;   // backoff, multiplied by attempt number

// Overall wall-clock budgets. Retries multiply worst-case latency (attempts x
// candidates x timeout), which is what made searches feel like they hung when
// StatCan was having a bad day. Once a budget is spent we stop retrying and
// work with whatever we already have.
const SEARCH_BUDGET_MS = 20000;
const DATA_BUDGET_MS   = 25000;

// Below this many provinces the choropleth is more misleading than useful.
// Matches the >= 8 province requirement /api/search uses when picking a cube.
const MIN_PROVINCES_FOR_MAP = 8;

// Transient = worth retrying. No response at all (connection reset / timeout /
// DNS), rate limiting, or a server-side error. Anything else is the request's
// own fault and will fail identically on a retry.
function isTransient(err) {
  const status = err.response?.status;
  if (status === undefined) return true;        // no response: network-level failure
  return status === 429 || status >= 500;
}

// POSTs to StatCan with bounded retries. Returns the axios response, or null if
// every attempt failed. NEVER throws — callers decide what a null means.
// `label` is only used for logging so failures are attributable in the server log.
async function statcanPost(endpoint, body, label, deadline = Infinity) {
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      return await axios.post(`${STATCAN}/${endpoint}`, body, {
        headers: { 'Content-Type': 'application/json' },
        timeout: REQUEST_TIMEOUT_MS
      });
    } catch (err) {
      const status    = err.response?.status;
      const transient = isTransient(err);
      const expired   = Date.now() >= deadline;
      const last      = attempt === MAX_ATTEMPTS || !transient || expired;

      console.warn(
        `${label}: attempt ${attempt}/${MAX_ATTEMPTS} failed ` +
        `(${err.code ?? err.message}${status ? ` http ${status}` : ''})` +
        (last
          ? ` — giving up${!transient ? ' (not retryable)' : expired ? ' (deadline reached)' : ''}.`
          : ' — retrying.')
      );

      if (last) return null;
      await new Promise(r => setTimeout(r, RETRY_BASE_MS * attempt));
    }
  }
  return null;
}

// Returns the cube metadata object, or null if it could not be retrieved or is
// unusable. NEVER throws — a rejection here used to unwind all the way to the
// /api/search handler's catch and turn one transient StatCan hiccup into a 500
// for the whole request. Callers treat null as "skip this candidate".
async function fetchMeta(cubeId, deadline) {
  const r = await statcanPost(
    'getCubeMetadata',
    [{ productId: parseInt(cubeId) }], // list of objects; could fetch several cubes at once later
    `Cube ${cubeId} metadata`,
    deadline
  );
  if (!r) return null;

  const entry = r.data?.[0];
  const meta  = entry?.object ?? null;

  // StatCan answers HTTP 200 with status:"FAILED" and `object` as a STRING for
  // withdrawn / non-existent product IDs, e.g.
  //   "The cube product ID 12345678 does not exist. Error code = CUBE_NOT_AVAILABLE"
  // That string is truthy, so a plain `if (!metadata)` guard downstream does not
  // catch it and `metadata.dimension.find(...)` throws a TypeError. Cubes go
  // missing between index rebuilds (925 of the indexed cubes are already
  // ARCHIVED), so this is expected traffic, not an exceptional case.
  if (typeof meta === 'string' || !Array.isArray(meta?.dimension)) {
    console.warn(
      `Cube ${cubeId}: unusable metadata (status=${entry?.status ?? 'unknown'}) — skipping. ` +
      (typeof meta === 'string' ? meta : 'no dimension array')
    );
    return null;
  }

  console.log(`Fetched metadata for cube ${cubeId}:`, meta.cubeTitleEn ?? 'No title found');
  return meta;
}


// Fetch a single coordinate for a single province.
//
// Returns a DISCRIMINATED result rather than a bare null, because the two
// failure modes need different handling and used to be indistinguishable:
//   { ok: true,  value, year }      - got a datapoint
//   { ok: false, reason: 'empty' }  - StatCan answered, but this cube genuinely
//                                     has no value at this coordinate
//   { ok: false, reason: 'failed' } - the request itself failed after retries
//
// Collapsing both into null is what let a flaky StatCan silently render a map
// with one province on it and no indication anything had gone wrong.
async function fetchCoordinate(cubeId, coordinate, deadline) {
  const r = await statcanPost(
    'getDataFromCubePidCoordAndLatestNPeriods',
    [{ productId: parseInt(cubeId), coordinate, latestN: 1 }], // only the latest period
    `Cube ${cubeId} coord ${coordinate}`,
    deadline
  );
  if (!r) return { ok: false, reason: 'failed' };

  // single point response; vectorDataPoint is an array, take the latest
  const pt = r.data?.[0]?.object?.vectorDataPoint?.[0];
  if (!pt || pt.value == null) return { ok: false, reason: 'empty' };

  const decimals = pt.decimals ?? 0;
  const value = decimals > 0
    ? Math.round(pt.value * 10 ** decimals) / 10 ** decimals
    : Number(pt.value);
  // returns year, gets rid of the '-' from the period string
  return { ok: true, value, year: pt.refPer?.split('-')[0] ?? 'N/A' };
}

// ── POST /api/search ──────────────────────────────────────────────────────────
// compares query embedding with the stored cube embeddings and finds the most semanticaly similar cubes
// then validates that the cube contains provincial data and extract the meta data about
// dimensions and provinces
/**
 * Request body: query, topK (optional, default 5)
 * Response: {}
 *  
 */
app.post('/api/search', async (req, res) => {
  try {
    const { query, topK = 5 } = req.body;
    if (!query?.trim()) return res.status(400).json({ error: 'Query is required' });

    console.log(`\n${'─'.repeat(60)}\nSearching: "${query}"`);
    const cubes = await loadCubes();
    const model = await getEmbeddingModel();
    const emb   = await model(query, { pooling: 'mean', normalize: true });
    const qVec  = Array.from(emb.data); // creates new array from the embedding data

    // console.log(`\n ${'-'.repeat(60)}$\n`);
    // console.log('qVec sample:', qVec.slice(0,5), 'qVec length:', qVec.length);
    
    // cube structure: { cubeId, title, embedding}
    // performs a cosine similary check between the query embedding and every
    // cube embedding (full list, not yet sliced to topK).
    const allScored = cubes
      .map(c => ({ cubeId: c.cubeId, title: c.title, similarity: cosineSimilarity(qVec, c.embedding) }))
      .sort((a, b) => b.similarity - a.similarity); // sort in descending order

    // Re-ranks the FULL scored list (not just the topK slice) using the MLP
    // query classifier: predicts a category for this query from qVec, then
    // boosts the similarity score of any cube whose mapped category matches
    // before re-sorting. Done before slicing so a cube just outside the raw
    // top-K on cosine similarity alone can still surface if its category
    // matches — slicing first would make the boost unable to pull in cubes
    // that didn't already make the cut.
    const { reranked, predictedCategory, predictedCategoryConfidence } =
      await rerankByCategory(allScored, query, qVec);

    if (predictedCategory) {
      console.log(
        `Predicted query category: ${predictedCategory} ` +
        `(confidence: ${(predictedCategoryConfidence * 100).toFixed(1)}%)`
      );
    }

    const ranked = reranked.slice(0, topK);
    
    console.log('Top results:');
    // logs the top K results with their similarity scores, truncated title to 60 characters and similarity percentage to 1 decimal place
    ranked.forEach((r, i) =>
      console.log(`  ${i+1}. [${r.cubeId}] ${r.title.slice(0,60)}… (${(r.similarity*100).toFixed(1)}%, category: ${r.cubeCategory ?? 'unknown'}, keyword matches: ${r.keywordMatches ?? 0})`)
    );
    // fetches the metadata for each of the top K cubes
    // Counts candidates we could not even evaluate because StatCan didn't answer,
    // so an upstream outage can be reported as 503 rather than being confused
    // with "we looked and nothing matched" (404). See the end of the loop.
    let unreachableCandidates = 0;
    const deadline = Date.now() + SEARCH_BUDGET_MS;

    for (const candidate of ranked) {
      const metadata = await fetchMeta(candidate.cubeId, deadline); // never throws; null = skip
      if (!metadata) {
        unreachableCandidates++;
        continue;
      }
      // locates the geography dimension by looking for keywords
      // (fetchMeta guarantees metadata.dimension is an array)
      const geoDim = metadata.dimension.find(d =>
        d.dimensionNameEn === 'Geography' || d.dimensionNameEn?.includes('Geography')
      );

      // does not consider cubes that do not have a geography dimension, since we need provincial data for the map
      // Array.isArray guard: StatCan occasionally returns a dimension with no
      // member list at all, which would throw on .filter below.
      if (!geoDim || !Array.isArray(geoDim.member)) continue;
      // removes members that are not provinces based on PROVINCE_MAPPING

      const provinces = geoDim.member // object with all the memebrs of the geography dimension
        .filter(m => PROVINCE_MAPPING[m.memberNameEn]) // transforms the member list into a list of objects with province name and memberId
        .map(m => ({ name: PROVINCE_MAPPING[m.memberNameEn], memberId: m.memberId })); // used in the /api/data endpoint below to build the coordinates

      // if there are less than 8 provinces, then the cube is skipped
      if (provinces.length < 8) continue; 

      let unit = null;

      const uomDim = metadata.dimension.find(d => d.hasUOM === true);
      // attempts to find the UoM
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
        // removes geography dimension since its handled separately
        .filter(({ idx }) => idx !== geoDimIndex)
        // transform each remaining dimension into example below:
        /** {
              name: "Time",
              dimIndex: 1,
              members: [...,
              ]
            },
            {
              name: "Age Group",
              dimIndex: 2,
              members: [ ...,
              ]
            }
          ];
         */
        // #TODO: If StatCan changes dimIndex to not be 0 based this would break
        .map(({ dim, idx }) => ({
          name:     dim.dimensionNameEn,
          dimIndex: idx,           
          members:  (dim.member ?? []) // use an empty array if there are no members
            .filter(m => m.memberId && m.memberId !== 0) // removes members with no memberId or memberId of 0 since memberId starts at 1
            .map(m => ({
              name:        m.memberNameEn,
              memberId:    m.memberId,
              isAggregate: isAggregate(m.memberNameEn),
            })),
        }));

      console.log(`dimensionMeta sample:`, dimensionMeta); 
      console.log(`dimensionMeta members sample:`, dimensionMeta[0]?.members[0]?.name);
      // padds it with leading zeros until it is 8 units long
      const pid     = String(candidate.cubeId).padStart(8, '0');
      // removes any hyphens from the cube ID
      const cleanPid = pid.replace(/-/g, '');

      // statcan table urls require a suffix like 01, if the cube already ends with 01 then don't do anything
      const fullPid = cleanPid.endsWith('01') ? cleanPid : `${cleanPid}01`;
      // full table URL
      const tableUrl = `https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=${fullPid}`;

      console.log(`Using cube ${candidate.cubeId}: ${candidate.title.slice(0, 60)}`);

      // Pulls in the richer fields from cubesMetadata.json that aren't part
      // of the slim cubesWithEmbeddings.*.json used for the similarity
      // search — footnotes, bilingual title, full dimension names/members,
      // subject/survey codes, archive status, series/datapoint counts.
      const extraMetadata = await enrichWithMetadata(candidate.cubeId);

      // Raw cosine similarity before category/keyword boosts — bounded [0,1]
      // so the frontend can display it as a clean 0–100% confidence figure.
      const matchConfidence = allScored.find(s => s.cubeId === candidate.cubeId)?.similarity ?? 0;

      return res.json({
        success: true,
        cubeId:       candidate.cubeId,
        title:        candidate.title,
        unit,
        tableUrl,
        geoDimIndex,
        provinces,
        dimensionMeta,
        predictedCategory,
        predictedCategoryConfidence,
        cubeCategory: candidate.cubeCategory ?? null,
        keywordMatches: candidate.keywordMatches ?? 0,
        isCurrent: isCurrentCube(extraMetadata),
        matchConfidence,
        ...extraMetadata,
      });
    }

    // Distinguish "StatCan was down for every candidate we tried" from "we
    // checked them all and none carry provincial data". Previously both a
    // transient outage and a genuine no-match produced the same message — and
    // an outage produced a 500 before that, since fetchMeta rethrew.
    if (unreachableCandidates === ranked.length && ranked.length > 0) {
      console.error(`All ${ranked.length} candidates unreachable — treating as upstream outage.`);
      return res.status(503).json({
        error: 'Statistics Canada is temporarily unavailable. Please try again in a moment.'
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
// Suppose a cube as dimensions: 1: geography, 2: time, 3: age group. 
// and each of those dimensions have members such as "Ontario" with memeberId 2 for geography,
// "2020" with memberId 10 for time, and "15-24" with memberId 1 for age group, each with their own memberId.
// Then 2, 10, 1, 0, 0, ... would give the row in the cube for Ontario, 15-24 age group, in 2020.
// returns a list of results where each result is { province: province.name, value: result.value, year: result.year }
app.post('/api/data', async (req, res) => {
  try {
    const { cubeId, geoDimIndex, provinces, selections } = req.body;

    if (!cubeId || !provinces?.length || !selections) {
      return res.status(400).json({ error: 'cubeId, provinces, and selections are required' });
    }

    console.log(`\nFetching data for cube ${cubeId}`);
    console.log(`Selections:`, selections);

    const results         = [];
    const failedProvinces = []; // request failed after retries — data may exist
    const emptyProvinces  = []; // StatCan answered: no value at this coordinate
    const deadline        = Date.now() + DATA_BUDGET_MS;

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
      const result = await fetchCoordinate(cubeId, coordinateStr, deadline);
      if (result.ok) {
        results.push({ province: province.name, value: result.value, year: result.year });
      } else if (result.reason === 'failed') {
        failedProvinces.push(province.name);   // request error — data may well exist
      } else {
        emptyProvinces.push(province.name);    // cube genuinely has no value here
      }
      // prevents too many API requests to StatCan in a short period of time.
      // StatCan limits servers to ~50 req/s, i.e. one per 20 ms — 10 ms was over
      // that ceiling and risked getting requests rate-limited (which then looked
      // like missing provinces on the map).
      await new Promise(r => setTimeout(r, 25));
    }

    if (failedProvinces.length) {
      console.error(
        `${failedProvinces.length}/${provinces.length} provinces failed to fetch ` +
        `(${failedProvinces.join(', ')}) — returning a partial result.`
      );
    }

    if (!results.length) {
      // Nothing came back at all. Which error depends on WHY: an upstream outage
      // is retryable and the user should be told to try again, whereas a cube
      // that simply has no data for this dimension combination is not.
      return failedProvinces.length
        ? res.status(503).json({ error: 'Statistics Canada is temporarily unavailable. Please try again in a moment.' })
        : res.status(404).json({ error: 'No data found for this combination' });
    }

    // A map drawn from one or two provinces is misleading, and silently drawing
    // it is what made this look like a data problem rather than a fetch problem.
    // Bail out only when the shortfall is due to failed requests — a cube that
    // genuinely only reports a few provinces is the cube's business, not an error.
    if (results.length < MIN_PROVINCES_FOR_MAP && failedProvinces.length) {
      return res.status(503).json({
        error:
          `Only ${results.length} of ${provinces.length} provinces could be retrieved ` +
          `from Statistics Canada. Please try again in a moment.`
      });
    }
    // takes only the year value from results
    const years = results.map(r => r.year).filter(y => y !== 'N/A');
    // sorts the year by frequency and takes the most common year among the results
    const year  = years.sort((a, b) =>
      years.filter(v => v === b).length - years.filter(v => v === a).length
    )[0] ?? 'N/A';

    console.log(
      `  → ${results.length}/${provinces.length} provinces returned` +
      (failedProvinces.length ? ` (${failedProvinces.length} fetch failures)` : '') +
      (emptyProvinces.length  ? ` (${emptyProvinces.length} with no data)`    : '')
    );
    // `incomplete` lets the frontend say why the map has gaps instead of
    // presenting a partial picture as though it were the whole story.
    return res.json({
      success: true,
      provinces: results,
      year,
      requestedProvinces: provinces.length,
      failedProvinces,
      emptyProvinces,
      incomplete: results.length < provinces.length
    });

  } catch (err) {
    console.error('Data fetch error:', err);
    res.status(500).json({ error: 'Internal server error', message: err.message });
  }
});
// check if the server is running and if the cubes are loaded in memory
app.get('/api/health', (_req, res) =>
  res.json({
    status: 'healthy',
    cubesLoaded: !!cachedCubes,
    embeddingModelLoaded: !!embeddingModel,
    mlpReady: mlpWarmedUp
  })
);

app.listen(PORT, () => console.log(`\nServer running on port ${PORT}`));
loadCubes().catch(console.error); // loads the cubes on server start

// Warms up the MLP query classifier at startup too, same reasoning as
// loadCubes() above — avoids the first /api/search request paying the cold
// model-load cost. Tracked separately from cachedCubes/embeddingModel since
// getMlpModel() caches internally inside Search_Utils.js; this flag is just
// for /api/health visibility.
getMlpModel()
  .then(() => { mlpWarmedUp = true; })
  .catch(err => {
    console.error(
      '❌ Failed to warm up MLP query classifier at startup — category ' +
      're-ranking will be unavailable until this succeeds:',
      err.message
    );
  });