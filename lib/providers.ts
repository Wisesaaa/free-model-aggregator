// The brief's two documents, the free model list, and the catalog.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) is prose with two JSON documents
// embedded in it, and the prose around them is irrelevant: this module locates each
// document by its `schema` member, under this contest's prefix
// (`free-model-aggregator-02/`).
//
//   - the free model list (`free-model-aggregator-02/models/v1`): one row per model,
//     keyed by the id the gateway routes (`openrouter/<vendor>/<model>:free`), with the
//     model's public rating in milli-points, what it can do (tools, vision, JSON mode,
//     context length) and its free flag. Only a free model may serve a prompt; this
//     module never routes a paid one.
//   - the six prompts (`free-model-aggregator-02/prompts/v1`): the prompts the house's
//     client will send, in order, each with its id, its api (`openai` or `anthropic`),
//     whether it streams, its weight (1 or 2) and the requirements a model must meet
//     to serve it. The client sends the bodies; this module routes each arrival by its
//     row, which is why the prompt list is read before the first request.
//
// A brief may spell the model list `models` or `ratings` and the prompts `prompts` or
// `manifest` (the sibling -01 contest's names), and one document may carry both; a
// document is located by the arrays it carries, never by a name alone. The brief's own
// prose is never parsed.
//
// Everything here is deterministic: no clock and no randomness, so `node main.ts --list`
// and the register body build the same document, byte for byte, on a live pass and on
// its replay.

import { readFileSync } from 'node:fs';

export const MODEL_LIST_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPT_LIST_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** Every schema this contest's brief writes carries this prefix; the brief's JSON is located by it. */
const CONTEST_SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches for them. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isJsonRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelEntry {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface ModelRequirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: ModelRequirement;
}

/** The brief as this module reads it. `error` is set when a document could not be read. */
export interface BriefDocument {
  models: ModelEntry[];
  prompts: PromptEntry[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModelEntry {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProviderEntry {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModelEntry[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogDocument {
  providers: CatalogProviderEntry[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
export interface ModelCapabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function matchingBraceAt(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** The JSON document with a string `schema` starting at `start`, or null when there is none there. */
function schemaDocumentAt(text: string, start: number): Json | null {
  const end = matchingBraceAt(text, start);
  if (end === -1) return null;
  try {
    const value: unknown = JSON.parse(text.slice(start, end + 1));
    if (isJsonRecord(value) && typeof value.schema === 'string') return value;
  } catch {
    // Not JSON on its own: nothing here.
  }
  return null;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function schemaDocumentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const doc = schemaDocumentAt(text, start);
    if (doc !== null) out.push(doc);
  }
  return out;
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asFiniteNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBoolean(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this contest's prefix that carries one of `keys` as an array: the document
 * and the key it was found under. A brief that spells its lists in the sibling's names still reads.
 */
function findArrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = asString(doc.schema);
    if (!schema.includes(CONTEST_SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * The gateway id for a bare model id: a row that gives the bare OpenRouter id
 * (`vendor/model:free`) and names `openrouter` is routed as the gateway spells it
 * (`openrouter/vendor/model:free`).
 */
function prefixedModelId(raw: string, named: string): string {
  return named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider that names it. The brief
 * lists free models, so a row is free unless it says `free: false`; a paid row is never routed.
 */
function modelRowOf(v: unknown): ModelEntry | null {
  if (!isJsonRecord(v)) return null;
  const raw = asString(v.id) !== '' ? asString(v.id) : asString(v.model);
  if (raw === '') return null;
  const named = asString(v.provider) !== '' ? asString(v.provider) : asString(v.wire);
  return {
    id: prefixedModelId(raw, named),
    provider: named === '' ? 'openrouter' : named,
    display_name: asString(v.display_name) !== '' ? asString(v.display_name) : raw,
    rating_milli: asFiniteNumber(v.rating_milli) ?? 0,
    context_length: asFiniteNumber(v.context_length),
    tools: asBoolean(v.tools),
    vision: asBoolean(v.vision),
    json: asBoolean(v.json),
    free: v.free !== false,
  };
}

/** The requirements a prompt row declares, keeping only the ones it states as true (or a finite context). */
function promptRequirementsOf(requires: Json): ModelRequirement {
  return {
    ...(requires.tools === true ? { tools: true } : {}),
    ...(requires.vision === true ? { vision: true } : {}),
    ...(requires.json === true ? { json: true } : {}),
    ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
  };
}

function promptRowOf(v: unknown): PromptEntry | null {
  if (!isJsonRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isJsonRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: promptRequirementsOf(requires),
  };
}

/** The brief file's text, or the reason it could not be read. */
function loadBriefText(path: string): { text: string; error: null } | { text: null; error: string } {
  try {
    return { text: readFileSync(path, 'utf8'), error: null };
  } catch (err) {
    return { text: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
}

/** The brief's two documents, found in `text` under this contest's schema prefix. */
function parseBriefDocuments(text: string, path: string): BriefDocument {
  const docs = schemaDocumentsIn(text);
  const modelsDoc = findArrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODEL_LIST_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelRowOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = findArrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPT_LIST_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptRowOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(path: string): BriefDocument {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  const loaded = loadBriefText(path);
  if (loaded.text === null) return { models: [], prompts: [], error: loaded.error };
  return parseBriefDocuments(loaded.text, path);
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModelsOf(brief: BriefDocument): ModelEntry[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOfRow(model: ModelEntry): ModelCapabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
export function satisfiesRequirements(requires: ModelRequirement, caps: ModelCapabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** The catalog's model rows: the brief's free models, each once, in the brief's order. */
function uniqueCatalogModels(models: readonly ModelEntry[]): CatalogModelEntry[] {
  const seen = new Set<string>();
  const out: CatalogModelEntry[] = [];
  for (const model of models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    out.push({ id: model.id, free: true, context_length: model.context_length });
  }
  return out;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves.
 *
 * The base_url is the provider's public endpoint, never the gateway's trial-local address: the gateway's
 * origin differs between a live pass and its replay, and the catalog is part of the state export a
 * replay must reproduce byte for byte.
 *
 * This baseline lists every free model in the brief — exactly the fourteen dialable ids the catalog
 * column scores — so keep it that way: register this set and keep GET /v1/models (both auth headers)
 * exactly that set, since listing fewer loses the column as surely as listing more.
 */
export function buildCatalogReport(brief: BriefDocument): CatalogDocument {
  const models = uniqueCatalogModels(brief.models);
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
