/* eslint-disable @typescript-eslint/no-require-imports -- Node's built-in test runner executes this CommonJS file. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');

const cache = new Map();
function loadTypeScript(relativePath) {
  const filename = path.resolve(__dirname, '..', relativePath);
  if (cache.has(filename)) return cache.get(filename).exports;
  const source = fs.readFileSync(filename, 'utf8');
  const javascript = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
  const loaded = { exports: {} };
  cache.set(filename, loaded);
  const localRequire = request => request.startsWith('.')
    ? loadTypeScript(path.join(path.dirname(relativePath), `${request}.ts`))
    : require(request);
  new Function('require', 'module', 'exports', 'process', javascript)(localRequire, loaded, loaded.exports, process);
  return loaded.exports;
}

const { presets } = loadTypeScript('src/lib/analysisTypes.ts');
const { mockModelInfoWire, createMockAnalysisWire } = loadTypeScript('src/lib/mockAnalysis.ts');
const { parseTemporalWindow, parseAnalysisResponse, parseModelInfoResponse, parseLumaeApiError, getModelInfo, analyzeVideo } = loadTypeScript('src/lib/lumaeApi.ts');
const { normalizeSaliency, saliencyPoints, selectPrediction, visibleTimelineWindows } = loadTypeScript('src/lib/analysisView.ts');

const videoMetadata = { durationSeconds: 30, width: 1280, height: 720, codec: 'h264' };
const sampleFile = new File(['sample'], 'sample.mp4', { type: 'video/mp4' });
const request = scenario => ({ video: { filename: sampleFile.name, ...videoMetadata }, queries: presets, scenario });

test('TemporalWindow parser accepts named fields and rejects tuples and invalid ranges', () => {
  assert.deepEqual(parseTemporalWindow({ start: 7.7, end: 15.46, confidence: 0.9275 }), { start: 7.7, end: 15.46, confidence: 0.9275 });
  assert.throws(() => parseTemporalWindow([7.7, 15.46, 0.9275]), error => error.code === 'MALFORMED_RESPONSE');
  assert.throws(() => parseTemporalWindow({ start: 5, end: 4, confidence: 0.5 }), error => error.code === 'MALFORMED_RESPONSE');
});

test('ModelInfoResponse parser owns the complete benchmark independently of analysis', () => {
  const info = parseModelInfoResponse(mockModelInfoWire);
  const backendManifest = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../backend/colab/model_manifest.json'), 'utf8'));
  assert.equal(parseModelInfoResponse(backendManifest).model.id, info.model.id);
  assert.equal(info.schemaVersion, '1.0');
  assert.equal(info.model.stage, 'baseline');
  assert.equal(info.benchmark.MR_R1_0_5, 53.23);
  assert.equal(info.benchmark.MR_mAP_0_5, 54.8);
  assert.equal(info.benchmark.HL_VeryGood_mAP, 35.51);
  assert.throws(() => parseModelInfoResponse({ ...mockModelInfoWire, benchmark: {} }), error => error.code === 'MALFORMED_RESPONSE');
  assert.throws(() => parseModelInfoResponse({ ...mockModelInfoWire, schema_version: '2.0' }), error => error.code === 'MALFORMED_RESPONSE');
});

test('AnalysisResponse parsing preserves raw saliency, nullable Hook, and model provenance', () => {
  const analysis = parseAnalysisResponse(createMockAnalysisWire(request('standard')));
  assert.equal(analysis.schemaVersion, '1.0');
  assert.equal(analysis.interpretation.hook.provenance, 'HEURISTIC');
  assert.equal(analysis.interpretation.hook.candidate, null);
  assert.equal(analysis.interpretation.demoMoment.provenance, 'MODEL_OUTPUT');
  assert.equal(analysis.interpretation.coreActionRegion.provenance, 'DERIVED');
  assert.equal(analysis.predictions[1].windows[0].confidence, 0.9275);
  assert.deepEqual(analysis.predictions[1].rawSaliencyScores.slice(0, 3), [-0.8359, -0.4944, 0.2634]);
  assert.equal('benchmark' in analysis, false);
  assert.throws(() => parseAnalysisResponse({ schema_version: '1.0', predictions: [] }), error => error.code === 'MALFORMED_RESPONSE');
});

test('normalization is display-only and handles constant scores', () => {
  const raw = [-0.8359, -0.4944, 0.2634];
  const before = [...raw];
  const normalized = normalizeSaliency(raw);
  assert.deepEqual(raw, before);
  assert.equal(normalized[0], 0);
  assert.equal(normalized[2], 1);
  assert.deepEqual(normalizeSaliency([-2, -2, -2]), [0, 0, 0]);
  const analysis = parseAnalysisResponse(createMockAnalysisWire(request('standard')));
  assert.equal(saliencyPoints(analysis.predictions[1])[1].raw, -0.4944);
  assert.equal(saliencyPoints(analysis.predictions[1])[1].time, 2);
});

test('query selection changes active prediction and timeline while benchmark stays model-level', () => {
  const info = parseModelInfoResponse(mockModelInfoWire);
  const analysis = parseAnalysisResponse(createMockAnalysisWire(request('standard')));
  assert.equal(selectPrediction(analysis, 'demo').windows[0].start, 7.7);
  assert.equal(selectPrediction(analysis, 'result').windows[0].start, 8);
  assert.equal(visibleTimelineWindows(analysis).length, 2);
  assert.equal(visibleTimelineWindows(analysis).some(item => item.prediction.queryId === 'hook'), false);
  assert.equal(info.benchmark.MR_mAP, 30.58);
});

test('mock scenarios use one validated API path', async () => {
  const info = await getModelInfo({ mode: 'mock' });
  assert.equal(info.model.id, 'moment-detr-clip-baseline');
  const standard = await analyzeVideo({ video: sampleFile, videoMetadata, queries: presets, mode: 'mock', scenario: 'standard' });
  assert.equal(standard.interpretation.hook.candidate, null);
  const hooked = await analyzeVideo({ video: sampleFile, videoMetadata, queries: presets, mode: 'mock', scenario: 'confident_hook' });
  assert.equal(hooked.interpretation.hook.candidate.confidence, 0.8421);
  assert.equal(visibleTimelineWindows(hooked).length, 3);
  const empty = await analyzeVideo({ video: sampleFile, videoMetadata, queries: presets, mode: 'mock', scenario: 'no_predictions' });
  assert.equal(empty.predictions.every(prediction => prediction.windows.length === 0), true);
  await assert.rejects(analyzeVideo({ video: sampleFile, videoMetadata, queries: presets, mode: 'mock', scenario: 'video_too_long' }), error => error.code === 'VIDEO_TOO_LONG');
  await assert.rejects(analyzeVideo({ video: sampleFile, videoMetadata, queries: presets, mode: 'mock', scenario: 'malformed_response' }), error => error.code === 'MALFORMED_RESPONSE');
});

test('API error schema and local validation have stable codes', async () => {
  assert.deepEqual(parseLumaeApiError({ schema_version: '1.0', error: { code: 'VIDEO_TOO_LONG', message: 'Too long', retryable: false } }).error.code, 'VIDEO_TOO_LONG');
  assert.throws(() => parseLumaeApiError({ schema_version: '1.0', error: { code: 'NEW_ERROR', message: 'x', retryable: false } }), error => error.code === 'MALFORMED_RESPONSE');
  await assert.rejects(analyzeVideo({ video: sampleFile, videoMetadata: { ...videoMetadata, durationSeconds: 151 }, queries: presets, mode: 'mock' }), error => error.code === 'VIDEO_TOO_LONG');
  await assert.rejects(analyzeVideo({ video: sampleFile, videoMetadata, queries: [], mode: 'mock' }), error => error.code === 'INVALID_QUERY');
});

test('real transport uses v1 URLs and multipart fields, then the same decoder', async () => {
  const oldFetch = global.fetch;
  const oldBase = process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL;
  const calls = [];
  process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL = 'https://example.test';
  global.fetch = async (url, init) => {
    calls.push({ url, init });
    return Response.json(url.endsWith('/model') ? mockModelInfoWire : createMockAnalysisWire(request('standard')));
  };
  try {
    await getModelInfo({ mode: 'real' });
    await analyzeVideo({ video: sampleFile, videoMetadata, queries: presets, mode: 'real' });
    assert.deepEqual(calls.map(call => call.url), ['https://example.test/api/v1/model', 'https://example.test/api/v1/analyze']);
    assert.equal(calls[1].init.body.get('video').name, 'sample.mp4');
    assert.equal(JSON.parse(calls[1].init.body.get('queries'))[1].id, 'demo');
  } finally {
    global.fetch = oldFetch;
    if (oldBase === undefined) delete process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL;
    else process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL = oldBase;
  }
});

test('real transport surfaces structured errors and malformed JSON without a mock fallback', async () => {
  const oldFetch = global.fetch;
  const oldBase = process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL;
  process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL = 'https://example.test';
  try {
    global.fetch = async () => Response.json({ schema_version: '1.0', error: { code: 'MODEL_UNAVAILABLE', message: 'Runtime offline', retryable: true } }, { status: 503 });
    await assert.rejects(getModelInfo({ mode: 'real' }), error => error.code === 'MODEL_UNAVAILABLE' && error.message === 'Runtime offline');
    global.fetch = async () => new Response('{ invalid json', { status: 200 });
    await assert.rejects(getModelInfo({ mode: 'real' }), error => error.code === 'MALFORMED_RESPONSE');
  } finally {
    global.fetch = oldFetch;
    if (oldBase === undefined) delete process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL;
    else process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL = oldBase;
  }
});

test('real transport reports offline and configurable timeout without mock fallback', async () => {
  const oldFetch = global.fetch;
  const oldBase = process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL;
  const oldTimeout = process.env.NEXT_PUBLIC_LUMAE_ANALYSIS_TIMEOUT_MS;
  process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL = 'https://example.test';
  process.env.NEXT_PUBLIC_LUMAE_ANALYSIS_TIMEOUT_MS = '5';
  try {
    global.fetch = async () => { throw new TypeError('network unavailable'); };
    await assert.rejects(getModelInfo({ mode: 'real' }), error => error.code === 'MODEL_UNAVAILABLE');
    global.fetch = (_url, init) => new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
    });
    await assert.rejects(analyzeVideo({ video: sampleFile, videoMetadata, queries: presets, mode: 'real' }),
      error => error.code === 'ANALYSIS_TIMEOUT');
  } finally {
    global.fetch = oldFetch;
    if (oldBase === undefined) delete process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL;
    else process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL = oldBase;
    if (oldTimeout === undefined) delete process.env.NEXT_PUBLIC_LUMAE_ANALYSIS_TIMEOUT_MS;
    else process.env.NEXT_PUBLIC_LUMAE_ANALYSIS_TIMEOUT_MS = oldTimeout;
  }
});
