# P0 implementation audit

Audited before the API v1 refactor.

| Category | Findings |
| --- | --- |
| KEEP | Next.js 16 App Router, npm/package-lock, strict TypeScript, ESLint, Tailwind 4 plus custom CSS, native MP4 upload and player, responsive three-column UI, timeline, saliency SVG, interpretation and loading/empty/error panels. No component library or global store exists; local React state is adequate. |
| REFACTOR | A single `page.tsx` held presentation plus data orchestration. UI consumed snake_case response fields and `[start,end,confidence]` tuples. Benchmark lived inside `AnalysisResponse`, and values `3.11`, `29.42`, `41.27` were duplicated in JSX. Playback cursor updated every `timeupdate`. Mock was imported directly into the page. |
| REMOVE | Automatic populated demo on initial load; implicit reliance on bundled mock despite no uploaded video; benchmark duplication in analysis; positional window API format. |
| MISSING | `GET /api/v1/model`, versioned analysis endpoint, schema version and model identity, standard error response, full runtime decoding, shared mock/real transport, scenario tests, semantics and API documents. |

Search results: `53.23`, `34.00`, `30.58`, `35.51`, and `55.87` were sourced from `src/lib/analysisTypes.ts` (plus tests); detail values were repeated in JSX. Temporal windows, saliency, and example confidence were centralized in `src/lib/mockAnalysis.ts`, not card components. There was one direct API `fetch` in `src/lib/lumaeApi.ts`, no `any` in core paths, no fake benchmark deltas, no destructive saliency normalization, and no forced Hook. The word “accuracy” occurred only in text saying benchmark metrics are not current-video accuracy.
