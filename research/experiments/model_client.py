"""Bedrock model client for the research experiments.

Registry-based multi-model client (per-region, per-tool-mode) so the ablation can sweep
several agent models via the AGENT_MODEL env var.

Account-specific settings (AWS profile, inference-profile ARNs) are NOT stored here. Put them in
model_registry_local.py next to this file (gitignored); it overrides the public defaults below.
Without it, the client uses public cross-region model IDs and the standard AWS credential chain
(AWS_PROFILE / environment variables / instance role).
"""
import ast, json, os, random, time
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError, ConnectionClosedError

# Long multi-hour sweeps hit Bedrock throttling; a single unhandled error used to abort the
# whole run. Retry transient failures with jittered exponential backoff.
# max_pool_connections: boto3 defaults to 10 connections per client, which silently caps
# concurrency at 10 in-flight calls no matter how many threads the experiment uses.
_BOTO_CFG = Config(read_timeout=300, connect_timeout=20, max_pool_connections=512,
                   retries={"max_attempts": 4, "mode": "standard"})
_RETRYABLE_CODES = {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
                    "ModelTimeoutException", "InternalServerException", "ModelNotReadyException",
                    "ModelErrorException"}
_MAX_CALL_ATTEMPTS = 10


RETRY_STATS = {"calls": 0, "retries": 0}


def _converse(region, **kwargs):
    """bedrock-runtime converse() with retry on throttling / transient network errors."""
    RETRY_STATS["calls"] += 1
    for attempt in range(_MAX_CALL_ATTEMPTS):
        try:
            return _client(region).converse(**kwargs)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in _RETRYABLE_CODES or attempt == _MAX_CALL_ATTEMPTS - 1:
                raise
        except (EndpointConnectionError, ReadTimeoutError, ConnectionClosedError):
            if attempt == _MAX_CALL_ATTEMPTS - 1:
                raise
        RETRY_STATS["retries"] += 1
        if RETRY_STATS["retries"] % 25 == 1:
            print(f"  [bedrock] retries so far: {RETRY_STATS['retries']} / calls {RETRY_STATS['calls']}", flush=True)
        time.sleep(min(120, (2 ** attempt) + random.uniform(0, 2)))

# Public defaults: system-defined cross-region model IDs (no account-specific data).
# Tuple: (model id or inference-profile ARN, region, tool_mode, supports_temperature)
# Claude 5 family (Sonnet 5 / Opus 5) are reasoning-tier and reject `temperature`.
REGISTRY = {
    "sonnet45":  ("eu.anthropic.claude-sonnet-4-5-20250929-v1:0",         "eu-west-1", "force", True),
    "sonnet5":   ("eu.anthropic.claude-sonnet-5",                         "eu-west-1", "force", False),
    "opus5":     ("eu.anthropic.claude-opus-5",                           "eu-west-1", "force", False),
    "haiku45":   ("eu.anthropic.claude-haiku-4-5-20251001-v1:0",          "eu-west-1", "force", True),
    "nova_pro":  ("eu.amazon.nova-pro-v1:0",                              "eu-west-1", "force", True),
    "nova_lite": ("eu.amazon.nova-lite-v1:0",                             "eu-west-1", "force", True),
    "llama4":    ("us.meta.llama4-maverick-17b-instruct-v1:0",            "us-west-2", "auto",  True),
    # Opus 5.5 rejects forced tool_choice on Bedrock -> "auto" mode with instruction retries.
    "opus55":       ("eu.anthropic.claude-opus-5-5",                      "eu-west-1", "auto",  False),
    "deepseek_v32": ("deepseek.v3.2",                                     "us-west-2", "force", True),
    "qwen3_235b":   ("qwen.qwen3-235b-a22b-2507-v1:0",                    "us-west-2", "force", True),
    "qwen3_32b":    ("qwen.qwen3-32b-v1:0",                               "us-west-2", "force", True),
    # Judges (cross-family presence checks only; never used for the core schema metric).
    "gpt6_astra":   ("us.openai.gpt-6-astra",                             "us-west-2", "force", False),
    "judge_oss":    ("openai.gpt-oss-120b-1:0",                           "eu-west-1", "force", True),
    "judge":        ("openai.gpt-oss-120b-1:0",                           "eu-west-1", "force", True),  # legacy alias
}
_PROFILE = os.environ.get("AWS_PROFILE")  # None -> default credential chain

try:  # local, gitignored overrides (AWS profile, account-specific inference-profile ARNs)
    import model_registry_local as _local
    REGISTRY.update(getattr(_local, "REGISTRY", {}))
    _PROFILE = getattr(_local, "AWS_PROFILE", _PROFILE)
except ImportError:
    pass

_clients = {}
def _client(region):
    if region not in _clients:
        session = boto3.Session(profile_name=_PROFILE) if _PROFILE else boto3.Session()
        _clients[region] = session.client("bedrock-runtime", region_name=region, config=_BOTO_CFG)
    return _clients[region]

AGENT_KEY = os.environ.get("AGENT_MODEL", "sonnet45")
AGENT_MODEL = AGENT_KEY
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt6_astra")
TEMPERATURE = 0.0

def _resolve(model_key):
    if model_key in REGISTRY:
        r = REGISTRY[model_key]
        # Backfill supports_temperature=True for legacy 3-tuple entries (defensive).
        return r if len(r) == 4 else (*r, True)
    return (model_key, "eu-west-1", "force", True)

MAX_TOKENS = 4096  # explicit generous cap so long consolidator belief-lists don't truncate mid-JSON
def _cfg(supports_temp):
    cfg = {"maxTokens": MAX_TOKENS}
    if supports_temp:
        cfg["temperature"] = TEMPERATURE
    return cfg

def complete_text(sp, up, model_id=None, seed=None):
    arn, region, _, supports_temp = _resolve(model_id or AGENT_MODEL)
    r=_converse(region, modelId=arn, system=[{"text":sp}],
        messages=[{"role":"user","content":[{"text":up}]}], inferenceConfig=_cfg(supports_temp))
    # Some models occasionally return an empty content array (safety filter, empty completion, etc.).
    # Prefer a text block if present; otherwise concatenate any text pieces found; fall back to "".
    content = r["output"]["message"].get("content", []) or []
    for b in content:
        if isinstance(b, dict) and "text" in b: return b["text"]
    return ""

def _inline_refs(sch):
    """Recursively resolve JSON-schema $ref pointers into their $defs targets and strip $defs.

    Reasoning-tier Anthropic models (Claude 5 family) on Bedrock's toolSpec do not reliably
    follow $ref indirection; the tool_use payload comes back missing the referenced fields.
    Inlining the schema keeps the toolSpec self-contained and works across all models tested.
    """
    defs = sch.get("$defs", {})
    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                if name in defs:
                    return resolve({k: v for k, v in defs[name].items() if k != "$defs"})
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node
    return resolve(sch)

def complete_structured(sp, up, pm, model_id=None, seed=None):
    """Structured completion. For models that reject forced toolChoice (mode='auto'),
    retry once with a stronger instruction if the first call doesn't emit a tool_use block."""
    arn, region, mode, supports_temp = _resolve(model_id or AGENT_MODEL)
    sch=_inline_refs(pm.model_json_schema())
    spec={"name":"return_structured_data","description":"Return the strictly formatted result. You MUST call this tool.",
          "inputSchema":{"json":{"type":"object","properties":sch.get("properties",{}),
          "required":sch.get("required",[])}}}
    tc={"tools":[{"toolSpec":spec}]}
    if mode=="force":
        tc["toolChoice"]={"tool":{"name":"return_structured_data"}}
        # Some models (e.g. DeepSeek V3.2) occasionally answer in prose despite forced
        # toolChoice. The first attempt uses the unmodified prompt, so any call that succeeds
        # first time behaves exactly as before; only refusals escalate to firmer instructions.
        sys_variants=[
            sp,
            sp+"\n\nYou MUST respond by calling the return_structured_data tool.",
            sp+"\n\nCRITICAL: Do NOT respond with free text or ask questions. Your ONLY valid output is a call to the return_structured_data tool, using the information given, even if it is short.",
        ]
    else:
        # try progressively more emphatic instructions; only relevant when the model won't be forced
        sys_variants=[
            sp+"\n\nYou MUST respond by calling the return_structured_data tool.",
            sp+"\n\nCRITICAL: Do NOT respond with free text. Your ONLY valid output is a call to the return_structured_data tool. No explanations, no commentary — just the tool call.",
        ]
    last = None
    parse_errs = []
    # Two retry axes: (a) sys_variants (stronger-instruction retries for auto-mode); (b) transient
    # response-degradation retries (Bedrock occasionally returns tool-use payload with fields as
    # stringified JSON under concurrent load; a plain retry with identical input usually recovers).
    max_transient_retries = 5
    for sys in sys_variants:
        got_toolUse_at_all = False
        for attempt in range(max_transient_retries + 1):
            r = _converse(region, modelId=arn, system=[{"text": sys}],
                messages=[{"role": "user", "content": [{"text": up}]}], toolConfig=tc, inferenceConfig=_cfg(supports_temp))
            tool_block = None
            for b in r["output"]["message"]["content"]:
                if "toolUse" in b:
                    tool_block = b; break
            if tool_block is None:
                last = r["output"]["message"]["content"]
                break  # no tool_use in response -> escalate to next sys_variant
            got_toolUse_at_all = True
            try:
                return pm(**_coerce(tool_block["toolUse"]["input"], pm))
            except Exception as pe:
                parse_errs.append((attempt, str(pe)[:80]))
                # transient retry: identical input, hope Bedrock returns a clean payload
                continue
        # An instruction variant produced no usable tool call (the next, firmer variant is tried if any); counted for reporting.
        RETRY_STATS["variant_failures"] = RETRY_STATS.get("variant_failures", 0) + 1
        if got_toolUse_at_all:
            # exhausted transient retries; try a stronger sys prompt if we have one
            continue
    # Name the model by its registry key, never by ARN (ARNs can carry account IDs into logs).
    raise ValueError(f"model '{model_id or AGENT_MODEL}' returned no valid tool use after {len(sys_variants)} instruction variants x {max_transient_retries+1} retries; parse_errs={parse_errs[:3]}; last={last if last is not None else '(no toolUse block)'}")

def _unwrap_schema_envelope(v):
    if isinstance(v, dict):
        if set(v.keys())=={"type","value"} and v["type"] in ("array","object"):
            return _unwrap_schema_envelope(v["value"])
        return {k:_unwrap_schema_envelope(x) for k,x in v.items()}
    if isinstance(v, list):
        return [_unwrap_schema_envelope(x) for x in v]
    return v

def _try_recover_truncated_array(s):
    """Sonnet occasionally returns a list field as a stringified JSON array that got truncated
    mid-element. Recover as many complete top-level {...} elements as we can parse.
    Returns a list on success, None on total failure."""
    s = s.strip()
    if not s.startswith("["):
        return None
    body = s[1:]
    items = []
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(body):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True; continue
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = body[start:i+1]
                try: items.append(json.loads(chunk))
                except Exception: pass
                start = None
    return items if items else None


def _coerce(data, pm):
    c = _unwrap_schema_envelope(dict(data))
    exp = set(pm.model_fields)
    if not (exp & c.keys()) and isinstance(c.get("properties"), dict):
        c = dict(c["properties"])
    if "confidence" in c:
        try:
            v = float(c["confidence"])
            if v > 1.0: c["confidence"] = v/100.0 if v <= 100 else 1.0
        except Exception: pass
    for fn, fi in pm.model_fields.items():
        v = c.get(fn)
        if isinstance(v, str) and v.strip().startswith(("[", "{")):
            parsed = None
            try: parsed = json.loads(v.strip())
            except Exception:
                try: parsed = ast.literal_eval(v.strip())
                except Exception:
                    # last resort: recover complete elements from a truncated JSON array
                    if v.strip().startswith("["):
                        parsed = _try_recover_truncated_array(v)
            if parsed is not None:
                c[fn] = parsed
    return c
