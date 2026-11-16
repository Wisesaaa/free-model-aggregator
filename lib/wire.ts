// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. The entry dials the grader's
// gateway, an OpenAI-compatible endpoint, so every prompt goes upstream as a Chat Completions request
// and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id) to the gateway and relays
//     the gateway's JSON or SSE answer, with every `model` field set to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the gateway's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really answered, so a response's
//   `model` must be the model this entry dialled. OpenRouter's own answer names a free model without
//   the gateway's `openrouter/` prefix; these functions put the dialled id back on every answer and
//   every streamed chunk. They never make a model up: the id came from the brief and went to the
//   gateway in the request's `model`.
//
// Everything here is a pure function of its arguments, except the stream relays: no clock and no
// randomness, so the same body and the same gateway answer give the same bytes on a replay. The two
// model listings use fixed timestamps for the same reason: a `created` that moved between runs would be
// a nondeterministic document.

import { isJsonRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelListing(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelListing(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicErrorBody(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIErrorBody(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textContentOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isJsonRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function contentPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isJsonRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultAsText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textContentOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsAsObject(value: unknown): Json {
  if (isJsonRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isJsonRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function anthropicStopReason(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** One Anthropic message as Chat Completions messages, in order: text, tool calls, tool results. */
function translateAnthropicMessage(raw: unknown): Json[] {
  if (!isJsonRecord(raw)) return [];
  const assistant = raw.role === 'assistant';
  const content = raw.content;
  if (typeof content === 'string') {
    return [{ role: assistant ? 'assistant' : 'user', content }];
  }
  if (!Array.isArray(content)) return [];
  if (assistant) {
    const toolCalls: Json[] = [];
    for (const block of content) {
      if (!isJsonRecord(block) || block.type !== 'tool_use') continue;
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isJsonRecord(block.input) ? block.input : {}),
        },
      });
    }
    const text = textContentOf(content);
    return [
      {
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      },
    ];
  }
  const parts: Json[] = [];
  const out: Json[] = [];
  for (const block of content) {
    if (!isJsonRecord(block)) continue;
    if (block.type === 'tool_result') {
      out.push({
        role: 'tool',
        tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        content: toolResultAsText(block.content),
      });
      continue;
    }
    const part = contentPartOf(block);
    if (part !== null) parts.push(part);
  }
  if (parts.length > 0) out.push({ role: 'user', content: parts });
  return out;
}

/** An Anthropic tools array as OpenAI function tools, skipping rows without a name. */
function translateAnthropicTools(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isJsonRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isJsonRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      },
    ];
  });
}

/** An Anthropic tool_choice as an OpenAI tool_choice value, or undefined when it names none. */
function translateAnthropicToolChoice(choice: unknown): Json | string | undefined {
  if (!isJsonRecord(choice)) return undefined;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'function', function: { name: choice.name } };
  return undefined;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicRequestAsOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textContentOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) messages.push(...translateAnthropicMessage(raw));

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  const tools = translateAnthropicTools(body.tools);
  if (tools.length > 0) out.tools = tools;
  const toolChoice = translateAnthropicToolChoice(body.tool_choice);
  if (toolChoice !== undefined) out.tool_choice = toolChoice;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function finiteNumberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer's tool calls as Anthropic tool_use blocks, in order. */
function toolUseBlocksOf(calls: readonly unknown[]): Json[] {
  const blocks: Json[] = [];
  for (const call of calls) {
    if (!isJsonRecord(call) || !isJsonRecord(call.function)) continue;
    blocks.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsAsObject(call.function.arguments),
    });
  }
  return blocks;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
export function openAIAnswerAsAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isJsonRecord(choices[0]) ? choices[0] : {};
  const message = isJsonRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textContentOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  content.push(...toolUseBlocksOf(calls));
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isJsonRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopReason(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: finiteNumberOrZero(usage.prompt_tokens), output_tokens: finiteNumberOrZero(usage.completion_tokens) },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
export function openAIJsonAsDialled(body: Json, model: string): Json {
  return { ...body, model };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function formatSseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One SSE data line with its payload's `model` set to the id the entry dialled; other lines pass through. */
function relabeledDataLine(line: string, model: string): string {
  const trimmed = line.trim();
  if (!trimmed.startsWith('data:')) return line;
  const payload = trimmed.slice(5).trim();
  if (payload === '' || payload === '[DONE]') return line;
  try {
    const chunk: unknown = JSON.parse(payload);
    if (isJsonRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
  } catch {
    // Not JSON: relay the line as it came.
  }
  return line;
}

/**
 * One SSE frame with every data payload's `model` set to `model`, and its blank line kept: comments,
 * `event:` lines and payloads that are not JSON objects pass through untouched. Chat Completions names
 * the model on its chunks; the client sees the id the entry dialled, never the provider's spelling.
 */
function frameWithDialledModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => relabeledDataLine(line, model));
  return `${lines.join('\n')}\n\n`;
}

/**
 * A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id the
 * entry dialled. Frames are split on the blank line that ends them, as the SSE grammar does; a frame cut
 * across reads is held until its end arrives, and a body that ends without one is relayed as it is.
 */
export async function* openAIStreamAsDialled(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield frameWithDialledModel(frame, model);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield frameWithDialledModel(buffer, model);
  } finally {
    reader.releaseLock();
  }
}

/** The mutable state of one Chat Completions stream being rewritten as Anthropic events. */
interface AnthropicStreamState {
  started: boolean;
  finished: boolean;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  pendingStop: string | null;
  textIndex: number;
  nextIndex: number;
  toolIndexes: Map<number, number>;
  inputTokens: number;
  outputTokens: number;
  /** Whether a text delta was emitted: a completion that produced text cost at least 1 output token. */
  sawText: boolean;
}

function newAnthropicStreamState(): AnthropicStreamState {
  return {
    started: false,
    finished: false,
    pendingStop: null,
    textIndex: -1,
    nextIndex: 0,
    toolIndexes: new Map(),
    inputTokens: 0,
    outputTokens: 0,
    sawText: false,
  };
}

/** The message_start event, once, on the first chunk that carries anything. */
function startMessageEvents(state: AnthropicStreamState, model: string): string[] {
  if (state.started) return [];
  state.started = true;
  return [
    formatSseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: state.inputTokens, output_tokens: 0 },
      },
    }),
  ];
}

/** The content_block_stop events for every block still open, text first, then tool calls in index order. */
function closeOpenContentBlocks(state: AnthropicStreamState): string[] {
  const out: string[] = [];
  if (state.textIndex !== -1) {
    out.push(formatSseEvent('content_block_stop', { type: 'content_block_stop', index: state.textIndex }));
    state.textIndex = -1;
  }
  for (const index of [...state.toolIndexes.values()].sort((a, b) => a - b)) out.push(formatSseEvent('content_block_stop', { type: 'content_block_stop', index }));
  state.toolIndexes.clear();
  return out;
}

/** The closing events: open blocks stopped, then message_delta with the stop reason and usage, then message_stop. */
function finishMessageEvents(state: AnthropicStreamState, stopReason: string): string[] {
  if (state.finished) return [];
  state.finished = true;
  return [
    ...closeOpenContentBlocks(state),
    // A completion that emitted text cost at least 1 output token, even when the stream's usage
    // chunk never arrived: the wire requires an output count of at least 1.
    formatSseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: state.outputTokens > 0 ? state.outputTokens : state.sawText ? 1 : 0 } }),
    formatSseEvent('message_stop', { type: 'message_stop' }),
  ];
}

/** One SSE frame's lines as Anthropic events, mutating the stream's state. */
function frameToAnthropicEvents(state: AnthropicStreamState, model: string, frame: string): string[] {
  const out: string[] = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '[DONE]') {
      out.push(...startMessageEvents(state, model), ...finishMessageEvents(state, state.pendingStop ?? 'end_turn'));
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isJsonRecord(chunk)) continue;
    if (isJsonRecord(chunk.usage)) {
      state.inputTokens = finiteNumberOrZero(chunk.usage.prompt_tokens);
      state.outputTokens = finiteNumberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isJsonRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isJsonRecord(choice.delta) ? choice.delta : {};
    out.push(...startMessageEvents(state, model));
    if (typeof delta.content === 'string' && delta.content !== '') {
      state.sawText = true;
      if (state.textIndex === -1) {
        state.textIndex = state.nextIndex;
        state.nextIndex += 1;
        out.push(formatSseEvent('content_block_start', { type: 'content_block_start', index: state.textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(formatSseEvent('content_block_delta', { type: 'content_block_delta', index: state.textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isJsonRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : state.toolIndexes.size;
      let index = state.toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (state.textIndex !== -1) {
          out.push(formatSseEvent('content_block_stop', { type: 'content_block_stop', index: state.textIndex }));
          state.textIndex = -1;
        }
        index = state.nextIndex;
        state.nextIndex += 1;
        state.toolIndexes.set(upstreamIndex, index);
        const fn = isJsonRecord(call.function) ? call.function : {};
        out.push(
          formatSseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isJsonRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(formatSseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
    // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) state.pendingStop = anthropicStopReason(choice.finish_reason);
  }
  return out;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the id the entry dialled, named back to the client.
 */
export async function* anthropicEventStream(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  const state = newAnthropicStreamState();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of frameToAnthropicEvents(state, model, frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of frameToAnthropicEvents(state, model, buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!state.finished) {
    for (const event of startMessageEvents(state, model)) yield event;
    for (const event of finishMessageEvents(state, state.pendingStop ?? 'end_turn')) yield event;
  }
}
