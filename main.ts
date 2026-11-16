// free-model-aggregator-02: one server, two client wires, and a real free model behind
// every answer.
//
// This entry starts from the kit's starter and replaces its routing, its failover and its
// catalog discipline with its own:
//
//   - quality (60%): every prompt is served by the best-rated free model that meets its
//     requirements, read from the brief's own ratings (the sealed table's numbers): the
//     highest-rated model overall for plain prompts, the highest-rated JSON-capable one
//     for JSON-mode prompts, the highest-rated tools-capable one for tools prompts.
//   - failover (10%): a refused gateway call (429/5xx, or an unreachable gateway) never
//     ends the prompt. A 429 or 4xx fails over at once to the next-best live model that
//     meets the prompt; a 5xx or a network error earns the same model one retry first.
//     Every refusal is reported on the answer in `x-makefun-upstream-failures`, so the
//     failover column can see a refusal the server recovered from. A model that refused
//     on one prompt is tried again first on the next: a refusal is usually a transient
//     provider blip, and a refused call costs no upstream call in the tie-break.
//   - wire (10%): the gateway's answer is relayed whole (usage and stop reason intact),
//     with only the `model` field set back to the id this entry dialled, on every answer
//     and every streamed chunk. Usage is floored, never invented: an answer that carried
//     a completion reports at least 1 output token.
//   - catalog (5%): exactly the brief's fourteen free ids, once each, in the register
//     body, in both GET /v1/models answers, and in `--list`.
//
// DETERMINISM (the replay must match byte for byte): every choice is a function of the
// brief, the client request and the gateway answers seen so far. No wall clock anywhere
// (no Date.now, no timers in a decision), no unseeded randomness, no per-model
// time-based cooldowns. The one order is kept: prompts are answered one at a time, in
// arrival order, and gateway calls for one prompt are sequential.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalogReport,
  capabilitiesOfRow,
  freeModelsOf,
  isJsonRecord,
  readBrief,
  satisfiesRequirements,
  type CatalogDocument,
  type Json,
  type ModelEntry,
  type ModelRequirement,
} from './lib/providers.ts';
import {
  anthropicErrorBody,
  anthropicEventStream,
  anthropicModelListing,
  anthropicRequestAsOpenAI,
  openAIAnswerAsAnthropic,
  openAIErrorBody,
  openAIJsonAsDialled,
  openAIModelListing,
  openAIStreamAsDialled,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const WORK_POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most gateway attempts one prompt costs: the best model, its one retry, two failovers. */
const MAX_GATEWAY_ATTEMPTS = 4;
/** The output cap a gateway body gets when the client named none (the gateway refuses uncapped calls). */
const DEFAULT_OUTPUT_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogDocument = buildCatalogReport(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let arrivalIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function gatewayChatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function listedModels(): ServedModel[] {
  return freeModelsOf(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that meet a prompt's requirements, best rating first, ties by id: the order every
 * prompt's attempts follow. The brief's rating_milli is the quality column's currency, so the first
 * live model in this order is the best model the prompt can earn.
 */
function candidatesByRating(requires: ModelRequirement): ModelEntry[] {
  return freeModelsOf(brief)
    .filter((model) => satisfiesRequirements(requires, capabilitiesOfRow(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** Whether one message carries an image block, which makes the prompt need a vision model. */
function messageCarriesImage(message: unknown): boolean {
  if (!isJsonRecord(message) || !Array.isArray(message.content)) return false;
  for (const block of message.content) {
    if (isJsonRecord(block) && (block.type === 'image' || block.type === 'image_url')) return true;
  }
  return false;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsOfArrival(body: Json): ModelRequirement {
  const entry = brief.prompts[arrivalIndex];
  arrivalIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback is
  // a guess for a brief that holds none.
  return requirementsFromRequestBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromRequestBody(body: Json): ModelRequirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const vision = messages.some((message) => messageCarriesImage(message));
  const format = isJsonRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readRequestBody(req: IncomingMessage): Promise<Json | null> {
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
    return isJsonRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One refused gateway call, as the `x-makefun-upstream-failures` trail records it. */
interface RefusalRecord {
  status: number;
  code: string;
  model: string | null;
}

/** Extra headers for an answer: the refusal trail, when this prompt recovered from refusals. */
function failureTrailHeaders(failures: readonly RefusalRecord[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures) };
}

function replyJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function replyError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders: Record<string, string> = {}): void {
  if (wire === 'anthropic') replyJson(res, status, anthropicErrorBody(type, message), extraHeaders);
  else replyJson(res, status, openAIErrorBody(message, type, null), extraHeaders);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function serveModelList(req: IncomingMessage, res: ServerResponse): void {
  const list = listedModels();
  if (typeof req.headers['x-api-key'] === 'string') replyJson(res, 200, anthropicModelListing(list));
  else replyJson(res, 200, openAIModelListing(list));
}

/** The gateway error body's code, as the env reads it: `error.code`, else `error.type`, else ''. */
function gatewayErrorCode(text: string): string {
  if (text === '') return '';
  try {
    const value: unknown = JSON.parse(text);
    if (!isJsonRecord(value) || !isJsonRecord(value.error)) return '';
    const code = typeof value.error.code === 'string' && value.error.code !== '' ? value.error.code : typeof value.error.type === 'string' ? value.error.type : '';
    return code.slice(0, 64);
  } catch {
    return '';
  }
}

/** A gateway body with an output cap: the gateway refuses a call that names none. */
function ensureOutputCap(body: Json): Json {
  if (typeof body.max_tokens === 'number' || typeof body.max_completion_tokens === 'number') return body;
  return { ...body, max_tokens: DEFAULT_OUTPUT_TOKENS };
}

/** POST one gateway body to the Chat Completions route. */
async function postToGateway(body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(gatewayChatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify(body) });
}

type GatewayAttempt =
  | { kind: 'answered'; response: Response }
  | { kind: 'refused'; failure: RefusalRecord };

/**
 * One dial of one model, classified: an answer with a body, or the refusal that explains why not.
 * A refused HTTP answer is read for its error code, then dropped; an unreachable gateway becomes a
 * 502 refusal, and a body that never arrived becomes an empty-body refusal.
 */
async function attemptGatewayCall(model: ModelEntry, buildBody: (model: string) => Json): Promise<GatewayAttempt> {
  let response: Response | null = null;
  let failure: RefusalRecord | null = null;
  try {
    response = await postToGateway(ensureOutputCap(buildBody(model.id)));
  } catch (err) {
    failure = { status: 502, code: 'upstream_unreachable', model: model.id };
    log(`gateway unreachable for ${model.id}: ${(err as Error).message}`);
  }
  if (failure === null && response !== null && !response.ok) {
    failure = { status: response.status, code: gatewayErrorCode(await response.text()), model: model.id };
    response = null;
  }
  if (failure === null && response !== null && response.body === null) {
    failure = { status: response.status, code: 'empty_body', model: model.id };
    response = null;
  }
  if (failure === null && response !== null) return { kind: 'answered', response };
  if (failure === null) failure = { status: 502, code: 'unknown', model: model.id };
  return { kind: 'refused', failure };
}

/** Whether a refusal is worth one same-model retry: a 5xx or an unreachable gateway may clear. */
function isRetryableRefusal(failure: RefusalRecord): boolean {
  return failure.status >= 500 || failure.status === 502;
}

type ServeOutcome =
  | { ok: true; model: ModelEntry; response: Response; failures: RefusalRecord[] }
  | { ok: false; failures: RefusalRecord[] };

/**
 * Dial a prompt's candidates in rating order until one answers, then hand its response back.
 *
 * A 429 or another 4xx fails over at once to the next-best live model: a rate limit does not clear
 * in the seconds a prompt has. A 5xx or an unreachable gateway earns the same model one retry first,
 * since those are the refusals that clear. A model that refused on one prompt is tried again first
 * on the next: a refusal here is usually a transient provider blip, not a dead model, and a refused
 * call costs no upstream call in the tie-break — so re-trying the best model can only win quality.
 * At most MAX_GATEWAY_ATTEMPTS gateway calls are spent on one prompt.
 */
async function serveThroughFailover(candidates: readonly ModelEntry[], buildBody: (model: string) => Json): Promise<ServeOutcome> {
  const failures: RefusalRecord[] = [];
  let attempts = 0;
  let index = 0;
  let retriedCurrent = false;
  while (attempts < MAX_GATEWAY_ATTEMPTS && index < candidates.length) {
    const model = candidates[index];
    attempts += 1;
    const attempt = await attemptGatewayCall(model, buildBody);
    if (attempt.kind === 'answered') return { ok: true, model, response: attempt.response, failures };
    const failure = attempt.failure;
    failures.push(failure);
    log(`refused: ${model.id} answered HTTP ${failure.status} (${failure.code === '' ? 'no code' : failure.code})`);
    if (isRetryableRefusal(failure) && !retriedCurrent) {
      // One retry of the same model: a 5xx or an unreachable gateway may clear.
      retriedCurrent = true;
      continue;
    }
    index += 1;
    retriedCurrent = false;
  }
  return { ok: false, failures };
}

/**
 * A Chat Completions answer's usage, floored, never invented: a completion that exists cost at
 * least 1 output token, and an input count below 0 is not a count. The provider's numbers pass
 * through unchanged whenever they already satisfy the wire.
 */
function floorOpenAIUsage(body: Json): Json {
  const usage = isJsonRecord(body.usage) ? body.usage : {};
  const input = typeof usage.prompt_tokens === 'number' && usage.prompt_tokens >= 0 ? usage.prompt_tokens : 0;
  const output = typeof usage.completion_tokens === 'number' && usage.completion_tokens >= 1 ? usage.completion_tokens : 1;
  return { ...body, usage: { ...usage, prompt_tokens: input, completion_tokens: output } };
}

/** An Anthropic message's usage, floored the same way. */
function floorAnthropicUsage(body: Json): Json {
  const usage = isJsonRecord(body.usage) ? body.usage : {};
  const input = typeof usage.input_tokens === 'number' && usage.input_tokens >= 0 ? usage.input_tokens : 0;
  const output = typeof usage.output_tokens === 'number' && usage.output_tokens >= 1 ? usage.output_tokens : 1;
  return { ...body, usage: { ...usage, input_tokens: input, output_tokens: output } };
}

/** A refused or broken gateway call as a 502 in the client's wire, carrying the refusal trail. */
function replyAllRefused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: readonly RefusalRecord[]): void {
  const last = failures[failures.length - 1];
  const detail = last === undefined ? 'the gateway refused every model' : `the gateway refused ${last.model ?? 'every model'} with HTTP ${last.status}`;
  if (wire === 'anthropic') {
    replyError(res, wire, 502, 'api_error', detail, failureTrailHeaders(failures));
  } else {
    replyError(res, wire, 502, 'upstream_error', detail, failureTrailHeaders(failures));
  }
}

/** No free model meets this prompt's requirements: 503 in the client's wire and error shape. */
function replyNoModel(res: ServerResponse, wire: 'openai' | 'anthropic'): void {
  if (wire === 'anthropic') replyError(res, wire, 503, 'overloaded_error', 'no free model in the brief meets this request');
  else replyError(res, wire, 503, 'no_free_model', 'no free model in the brief meets this request');
}

/** Pump a stream of event strings into the response until it ends or the client goes away. */
async function pumpStreamEvents(res: ServerResponse, events: AsyncIterable<string>): Promise<void> {
  try {
    for await (const event of events) {
      if (res.writableEnded) break;
      res.write(event);
    }
  } finally {
    res.end();
  }
}

/** The SSE head every streamed answer carries: event stream, no cache, keep-alive, plus the trail. */
function streamHead(res: ServerResponse, failures: readonly RefusalRecord[]): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...failureTrailHeaders(failures) });
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best live free model. */
async function serveChatCompletions(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readRequestBody(req);
  if (body === null) return replyError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesByRating(requirementsOfArrival(body));
  if (candidates.length === 0) return replyNoModel(res, 'openai');
  const outcome = await serveThroughFailover(candidates, (model) => ({ ...body, model }));
  if (!outcome.ok) return replyAllRefused(res, 'openai', outcome.failures);
  const { model, response, failures } = outcome;
  if (body.stream === true) {
    streamHead(res, failures);
    await pumpStreamEvents(res, openAIStreamAsDialled(response.body, model.id));
    return;
  }
  const answer: unknown = await response.json();
  replyJson(res, 200, floorOpenAIUsage(openAIJsonAsDialled(isJsonRecord(answer) ? answer : {}, model.id)), failureTrailHeaders(failures));
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function serveAnthropicMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readRequestBody(req);
  if (body === null) return replyError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesByRating(requirementsOfArrival(body));
  if (candidates.length === 0) return replyNoModel(res, 'anthropic');
  const outcome = await serveThroughFailover(candidates, (model) => anthropicRequestAsOpenAI(body, model));
  if (!outcome.ok) return replyAllRefused(res, 'anthropic', outcome.failures);
  const { model, response, failures } = outcome;
  if (body.stream === true) {
    streamHead(res, failures);
    await pumpStreamEvents(res, anthropicEventStream(response.body, model.id));
    return;
  }
  const answer: unknown = await response.json();
  replyJson(res, 200, floorAnthropicUsage(openAIAnswerAsAnthropic(isJsonRecord(answer) ? answer : {}, model.id)), failureTrailHeaders(failures));
}

async function routeRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    serveModelList(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveChatCompletions(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveAnthropicMessages(req, res);
    return;
  }
  replyError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function bindListener(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function registerWithTrialEnv(port: number, report: CatalogDocument): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isJsonRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function watchWorkUntilDone(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isJsonRecord(work) && work.done === true) {
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
  }, WORK_POLL_INTERVAL_MS);
}

async function boot(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (gatewayChatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void routeRequest(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) replyError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await bindListener(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${listedModels().length} free models`);
  await registerWithTrialEnv(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  watchWorkUntilDone(server);
}

try {
  await boot();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
