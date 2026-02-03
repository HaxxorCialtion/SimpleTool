# SimpleTool Game Development Guide

> Reference document for AI assistants when generating SimpleTool-based games

## 1. Overview

SimpleTool is a **Multi-Head Parallel Decoding** acceleration framework for LLM function calling. Unlike traditional autoregressive decoding, it uses vLLM batch inference to generate multiple output heads simultaneously, reducing latency from 200-500ms to 25-50ms.

```
Traditional FC:  function → arg1 → arg2 → ... (sequential, O(n))
SimpleTool:      [function, arg1, arg2, ...] (parallel, O(1))
```

## 2. Server API

### Endpoints
- `GET /health` - Health check
- `POST /v1/function_call` - Standard function call

### Request Format
```javascript
{
  messages: [{role: 'user', content: 'your query'}],
  tools: [...],                // OpenAI-format tool definitions
  environment: [...],          // Current state info (string array)
  history: [...],              // Action history (string array, max 6)
  include_content_head: false  // Whether to generate content head
}
```

### Response Format
```javascript
{
  success: true,
  function: "move",
  args: {direction: "up"},
  heads: {
    function: "move",
    arg1: "up",
    arg2: "<|null|>",
    // ...
  },
  latency_ms: 35.2
}
```

## 3. Dynamic Head Count (Important!)

### Core Principle
**Do NOT always spawn 8 heads.** The server dynamically selects active heads based on:

1. **Content head** - Only include if `include_content_head: true` (usually unnecessary for games)
2. **Function head** - Always required
3. **Argument heads** - Only spawn heads for the maximum argument count across all tools

### Example
If your tools have at most 2 parameters:
```javascript
// Tools definition
const TOOLS = [{
  type: "function",
  function: {
    name: "move",
    parameters: {
      properties: {
        direction: {...},  // arg1
        speed: {...}       // arg2
      }
    }
  }
}];

// Server will only spawn 3 heads: <function>, <arg1>, <arg2>
// NOT 8 heads - this saves ~40% latency
```

### Head Selection Logic
```
Active Heads = [<function>] + [<arg1>...<argN>] where N = max(tool_param_counts)

If include_content_head:
  Active Heads = [<content>] + Active Heads
```

## 4. Tool Definition

### Constraints
- **Maximum 6 arguments** (arg1-arg6)
- Arguments map to arg1, arg2, ... in definition order
- Server auto-converts types: numeric string → int, then try float, else string

### Recommended Pattern
```javascript
const TOOLS = [{
  type: "function",
  function: {
    name: "action_name",
    description: "Clear, concise description",
    parameters: {
      type: "object",
      properties: {
        param1: {
          type: "string",
          enum: ["opt1", "opt2"],  // Use enum to constrain options
          description: "What this param does"
        }
      },
      required: ["param1"]
    }
  }
}];
```

## 5. Query Design

### Principles
1. **Give clear instructions** - Tell the model what to do, not just describe state
2. **Include decision hints** - e.g., "Move DOWN to intercept" not "Ball is below"
3. **List available options** - e.g., "Choose: up/down/stay"

### Good vs Bad Examples

✅ **Good Query**
```
Ball is 50px BELOW paddle. Move DOWN to intercept. Choose: up/down/stay
```

❌ **Bad Query**
```
Ball position: 250, Paddle position: 200. What should I do?
```

### Environment Design
- Use `key=value` format
- Keep it minimal - only decision-critical info
- Use semantic labels (e.g., `aligned=true`)

```javascript
const env = [
  `target_y=${targetY}`,
  `self_y=${selfY}`,
  `diff=${diff}`,
  `approaching=true`
];
```

## 6. Frontend Code Standards

### Required: Type-Safe Value Extraction
```javascript
// ✅ Correct - Force string conversion
function safeStr(v) {
  if (v === null || v === undefined) return '';
  return String(v).trim().toLowerCase();
}

let direction = safeStr(d.args?.direction) || safeStr(d.heads?.arg1);

// ❌ Wrong - .trim() may fail if v is int
let direction = (d.args?.direction || '').trim();
```

### Required: Validate Return Values
```javascript
const VALID_DIRECTIONS = ['up', 'down', 'left', 'right'];

if (!VALID_DIRECTIONS.includes(direction)) {
  console.warn(`Invalid direction: "${direction}", using fallback`);
  direction = 'stay';
}
```

### Required: Logging
```javascript
const log = (msg, type = 'info') => {
  console.log(`[GameName][${type}] ${msg}`);
};

log(`Query: ${query}`);
log(`Decision: ${direction} in ${latency.toFixed(0)}ms`, 'success');
log(`Invalid response: ${JSON.stringify(raw)}`, 'error');
```

### Required: Error Handling with Fallback
```javascript
async function callAI() {
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    // Process normal response
  } catch (e) {
    console.error('[AI] Call failed:', e);
    applyFallbackAI();  // Must have fallback logic
  }
}
```

### Recommended: Debug UI
Display in your game UI:
- Current environment state
- Sent query
- Raw response
- Latency statistics (current, average)

## 7. Game Loop Pattern

```javascript
const AI_INTERVAL = 100;  // 100ms = 10Hz

let aiPending = false;

// Main game loop (60fps)
function gameLoop() {
  update();
  render();
  requestAnimationFrame(gameLoop);
}

// AI decision loop (separate from render)
async function aiLoop() {
  if (aiPending) return;
  aiPending = true;
  await callAI();
  aiPending = false;
}

setInterval(aiLoop, AI_INTERVAL);
```

## 8. FCClient Template

```javascript
class FCClient {
  constructor(url) {
    this.url = url.replace(/\/$/, '');
  }

  async health() {
    try {
      const r = await fetch(`${this.url}/health`, {
        signal: AbortSignal.timeout(3000)
      });
      const d = await r.json();
      return { ok: d.loaded === true || d.status === 'ok' };
    } catch (e) {
      return { ok: false };
    }
  }

  async call(query, env, history, tools, includeContent = false) {
    const t0 = performance.now();
    try {
      const r = await fetch(`${this.url}/v1/function_call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: query }],
          tools,
          environment: env,
          history,
          include_content_head: includeContent
        })
      });
      const d = await r.json();
      return { ...d, ms: performance.now() - t0 };
    } catch (e) {
      return { success: false, error: e.message, ms: performance.now() - t0 };
    }
  }
}
```

## 9. Troubleshooting

### AI behaves erratically (stuck, no movement)
Check:
1. Is the query clear enough?
2. Is the returned value valid?
3. Are you extracting from `args` correctly (with `heads` as fallback)?

### `.trim is not a function` error
Cause: Values in `args` may be int, not string.
Fix: Use `String(v)` for type coercion.

### High latency
Check:
1. Is prefix caching enabled on server?
2. Is query/environment too long?
3. Network latency?
4. Are you spawning unnecessary heads?

---

**Version**: 1.0  
**Last Updated**: 2025-02
