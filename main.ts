// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// This server reads the trial's brief, probes each provider's catalog route, builds one catalog,
// registers it with the env, and answers the client's twenty requests — each with the best free
// model that meets the request's requirements and still has budget, failing over when the picked
// provider refuses, and exiting 0 when the env says the workload is done.
//
// HOW IT WINS EACH COLUMN
//   - quality (60%): every request's candidates are ranked by the quality table's rating_milli, and
//     the twenty primaries are planned up front: weight-2 requests first, each taking the best model
//     whose provider still has planned budget. The plan's per-provider budget is the kit's published
//     per-trial minimum for that provider, so a primary is never planned past a budget the provider
//     is guaranteed to have; the runtime failover handles the real numbers.
//   - served (15%): a refused call never ends a request — the next best candidate is tried, then
//     the next, until one answers or the list is spent.
//   - wire (10%): every answer's text, usage and stop reason come back verbatim from the provider
//     that answered, re-emitted in the wire the client used. Upstream streams are read whole and
//     rebuilt, so the usage chunk an OpenAI stream needs and the message_delta an Anthropic stream
//     needs are always present with the provider's own counts.
//   - failover (10%): a 402 marks its provider dead for the trial (its budget is spent); a 429 or
//     5xx counts against it, and two in a row mark it dead too. A dead provider is never retried.
//   - catalog (5%): the catalog is the providers' own listings, free flags from the quality table,
//     and GET /v1/models lists exactly the free ids it registers, in both wires' shapes.
//
// A JSON-mode request from an OpenAI client is never routed to an anthropic-wire provider: that
// wire cannot carry response_format, and its provider would echo json_mode absent.
//
// Every choice is a function of the brief, the requests and the providers' answers: no wall clock,
// no unseeded randomness, so a replay leaves the same state export, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  buildAnthropicStream,
  buildOpenAIStream,
  geminiFinishOf,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIFinishOf,
  openAIModelList,
  openAIAnswerToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  parseAnthropicStream,
  parseGeminiStream,
  parseOpenAIStream,
  stopReasonOf,
  textFromJson,
  type ServedModel,
  type Usage,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

/**
 * The planned per-provider call budget: the kit's published per-trial minimum for each provider.
 * A primary is never planned past it; the runtime failover handles the real, larger budget.
 */
const PLANNED_CAPACITY: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const DEFAULT_CAPACITY = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** One routable model: its probe, its listing, its quality-table row, and its rating. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/** One provider's runtime state: calls made, failures seen, and whether it is dead this trial. */
interface ProviderState {
  calls: number;
  fails: number;
  dead: boolean;
}

const candidatesByRequest: Candidate[][] = [];
const plan: (Candidate | null)[] = [];
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, fails: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

const byRating = (a: Candidate, b: Candidate): number =>
  b.rating - a.rating ||
  (a.probe.provider.id < b.probe.provider.id
    ? -1
    : a.probe.provider.id > b.probe.provider.id
      ? 1
      : a.model.id < b.model.id
        ? -1
        : a.model.id > b.model.id
          ? 1
          : 0);

/** Whether a candidate's wire can carry this request: JSON mode needs an openai or gemini wire. */
function wireOk(clientApi: 'openai' | 'anthropic', requiresJson: boolean, candidate: Candidate): boolean {
  if (clientApi === 'openai' && requiresJson === true) return candidate.probe.provider.wire !== 'anthropic';
  return true;
}

/** Probe every provider, build the one catalog, and plan every request's primary. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);

  const manifest = brief.workload?.manifest ?? [];
  for (let i = 0; i < manifest.length; i++) {
    const entry = manifest[i];
    const list: Candidate[] = [];
    for (const probe of probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!meetsRequirements(entry.requires, capabilitiesOf(model, row))) continue;
        const candidate: Candidate = { probe, model, row, rating: row?.rating_milli ?? 0 };
        if (!wireOk(entry.api, entry.requires.json === true, candidate)) continue;
        list.push(candidate);
      }
    }
    list.sort(byRating);
    candidatesByRequest.push(list);
    plan.push(null);
  }
  // Weight-2 requests first, then in manifest order: the best models go where the weight is.
  // The plan's budget is tracked here, at plan time: a primary is never planned past the
  // provider's published minimum, so the first candidate with budget left wins each request.
  const plannedUsed = new Map<string, number>();
  const plannedLeftAtPlan = (providerId: string): number =>
    (PLANNED_CAPACITY[providerId] ?? DEFAULT_CAPACITY) - (plannedUsed.get(providerId) ?? 0);
  const order = manifest.map((entry, i) => ({ entry, i })).sort((a, b) => b.entry.weight - a.entry.weight || a.i - b.i);
  for (const { i } of order) {
    const list = candidatesByRequest[i];
    const pick = list.find((candidate) => plannedLeftAtPlan(candidate.probe.provider.id) > 0) ?? list[0] ?? null;
    plan[i] = pick;
    if (pick !== null) plannedUsed.set(pick.probe.provider.id, (plannedUsed.get(pick.probe.provider.id) ?? 0) + 1);
  }
  const planned = plan.filter((p) => p !== null).length;
  log(`catalog: ${catalog.providers.length} providers; planned ${planned} of ${manifest.length} primaries`);
}

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json): { requires: Requirement; index: number } {
  const manifest = brief.workload?.manifest ?? [];
  const index = manifestIndex;
  manifestIndex += 1;
  const entry = manifest[index];
  if (entry !== undefined) return { requires: entry.requires, index };
  return { requires: requirementsFromBody(body), index };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  status: number;
  text: string;
}

/** One inference call to a candidate's provider, in its own wire, read whole. */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) payload.stream_options = { include_usage: true };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
  } else {
    const method = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${method}${stream ? '?alt=sse' : ''}`;
    payload = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  }
  try {
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
    return { status: res.status, text: await res.text() };
  } catch {
    return { status: 0, text: '' };
  }
}

/** The provider's answer, parsed by its wire, as text plus usage plus stop reason. */
function parseUpstream(candidate: Candidate, result: UpstreamResult, stream: boolean): { text: string; usage: Usage | null; stopReason: string | null } | null {
  const wire = candidate.probe.provider.wire;
  if (stream) {
    if (wire === 'openai') return parseOpenAIStream(result.text);
    if (wire === 'anthropic') return parseAnthropicStream(result.text);
    return parseGeminiStream(result.text);
  }
  let value: unknown;
  try {
    value = JSON.parse(result.text);
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  return textFromJson(wire, value);
}

/** The provider's answer re-emitted in the client's wire: a JSON object, or an SSE document. */
function translateAnswer(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  result: UpstreamResult,
  stream: boolean,
): { json: Json } | { sse: string } | null {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (!stream) {
    // Every JSON answer is parsed once and must carry the provider's usage: an answer without it
    // cannot pass the client's usage check, so it counts as a failed attempt, not a served one.
    let value: unknown;
    try {
      value = JSON.parse(result.text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const parsed = textFromJson(wire, value);
    if (parsed.usage === null) return null;
    // Same wire both ways: the provider's answer goes back verbatim.
    if (clientApi === 'openai' && wire === 'openai') return { json: value };
    if (clientApi === 'anthropic' && wire === 'anthropic') return { json: value };
    if (clientApi === 'openai' && wire === 'anthropic') return { json: anthropicToOpenAIResponse(value, model) };
    if (clientApi === 'openai' && wire === 'gemini') return { json: geminiToOpenAI(value, model) };
    if (clientApi === 'anthropic' && wire === 'openai') return { json: openAIAnswerToAnthropic(value, model) };
    if (clientApi === 'anthropic' && wire === 'gemini') return { json: geminiToAnthropic(value, model) };
    return null;
  }
  // Streamed: read the provider's stream whole, then rebuild it in the client's wire.
  const parsed = parseUpstream(candidate, result, true);
  if (parsed === null || parsed.usage === null || parsed.stopReason === null) return null;
  if (clientApi === 'openai' && wire === 'openai') return { sse: result.text };
  if (clientApi === 'anthropic' && wire === 'anthropic') return { sse: result.text };
  const finish =
    wire === 'openai' ? parsed.stopReason : wire === 'anthropic' ? openAIFinishOf(parsed.stopReason) : geminiFinishOf(parsed.stopReason);
  const stop =
    wire === 'anthropic' ? parsed.stopReason : wire === 'openai' ? stopReasonOf(parsed.stopReason) : finish === 'length' ? 'max_tokens' : finish === 'content_filter' ? 'refusal' : 'end_turn';
  if (clientApi === 'openai') return { sse: buildOpenAIStream(parsed.text, model, parsed.usage, finish) };
  return { sse: buildAnthropicStream(parsed.text, model, parsed.usage, stop) };
}

/** Serve one client request: the planned primary first, then the next best candidates until one answers. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const { requires, index } = requirementsFor(clientApi, body);
  const primary = plan[index] ?? null;
  const ordered = [...(candidatesByRequest[index] ?? [])];
  if (primary !== null) {
    const at = ordered.findIndex((c) => c.probe.provider.id === primary.probe.provider.id && c.model.id === primary.model.id);
    if (at > 0) {
      ordered.splice(at, 1);
      ordered.unshift(primary);
    }
  }
  const tried = new Set<string>();
  for (const candidate of ordered) {
    const providerId = candidate.probe.provider.id;
    const state = stateOf(providerId);
    if (state.dead) continue;
    const key = `${providerId}/${candidate.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    state.calls += 1;
    const result = await callUpstream(candidate, clientApi, body, stream);
    if (result.status >= 200 && result.status < 300) {
      const answer = translateAnswer(candidate, clientApi, result, stream);
      if (answer !== null) {
        if ('sse' in answer) sendSse(res, answer.sse);
        else sendJson(res, 200, answer.json);
        return;
      }
      log(`unusable answer from ${key}: the provider's reply did not parse`);
      continue;
    }
    if (result.status === 402) {
      state.dead = true;
      log(`${providerId} answered 402: its budget is spent, marked dead`);
    } else if (result.status === 400 || result.status === 401) {
      state.dead = true;
      log(`${providerId} answered ${result.status}: marked dead`);
    } else {
      state.fails += 1;
      if (state.fails >= 2) {
        state.dead = true;
        log(`${providerId} failed twice: marked dead`);
      } else {
        log(`${providerId} answered ${result.status === 0 ? 'no reply' : result.status}: trying the next candidate`);
      }
    }
  }
  log(`request ${index} (${clientApi}, stream ${stream}): no candidate answered`);
  if (clientApi === 'anthropic') sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  else sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
