# pip install --upgrade openai tqdm
import os, json, time, math, traceback
from typing import List, Dict, Any, Iterable, Optional
from tqdm import tqdm

# --- OpenAI-compatible client (llm7) ---
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

LLM7_BASE_URL = "https://api.llm7.io/v1"
LLM7_MODEL = "gpt-4o-mini-2024-07-18"

client = OpenAI(base_url=LLM7_BASE_URL, api_key="unused")

# ===== Prompts (keep your full texts if you trimmed earlier) =====
INITIAL_ANALYSIS_PROMPT = """You are an expert aviation accident analyst...
Analyze THIS ONE accident record and determine whether situational awareness (SA) loss was a contributing factor.

Your answer must have:
1) Conclusion: "Yes" or "No".
2) Reasoning: step-by-step, citing details FROM THE TEXT below. If "Yes", name likely SA-loss drivers (e.g., distraction, channelized attention, overload, fatigue, weather). at the end write a list of causes of SA loss (just with one word for each.
--- ACCIDENT RECORD ---
{report_text}
"""

SELF_CRITIQUE_PROMPT = """You are a QA reviewer. Critique the analysis strictly using ONLY the given record text.

--- RECORD TEXT ---
{report_text}

--- INITIAL ANALYSIS ---
{initial_analysis}

Checklist:
1) Evidence Check: Are all claims supported by the record? Flag unsupported ones.
2) Alternatives: Did it miss other plausible causes/mechanisms stated/implied in the record?
3) Conclusion Strength: Is the Yes/No warranted by the evidence?
4) Bias: Any assumptions beyond the text?

Provide a structured critique.
"""

FINAL_REVISION_PROMPT = """Revise the initial analysis using the critique. Keep the two-part structure:
- Conclusion (Yes/No)
- Reasoning (with explicit citations to the record text)

--- RECORD TEXT ---
{report_text}

--- INITIAL ANALYSIS ---
{initial_analysis}

--- CRITIQUE ---
{critique_text}

Now output the final improved analysis.
"""

# ===== Safer LLM caller with retries/backoff and token guards =====
def call_llm(messages: List[Dict[str, str]],
             temperature: float = 0.3,
             max_tokens: int = 900,
             max_retries: int = 6,
             base_delay: float = 1.0) -> str:
    """
    Robust call with retries on transient/server errors.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LLM7_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,  # seconds
            )
            return resp.choices[0].message.content.strip()
        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            # transient/network/rate
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
        except APIStatusError as e:
            # 5xx => retry; 4xx => raise
            status = getattr(e, "status_code", None)
            if status in {500, 502, 503, 504}:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                # Likely a bad request; bubble up.
                raise
        except Exception:
            # Unknown error; retry a couple of times then raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    # If we got here, retries exhausted
    raise RuntimeError("LLM request failed after retries.")

# ===== Helpers to keep requests small =====
def clamp_text(text: str, max_chars: int = 12000) -> str:
    """Truncate text to avoid overlong payloads."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    note = f"\n\n[TRUNCATED: original length {len(text)} chars]"
    return head + note

PREFERRED_KEYS = [
    # Common fields seen in NTSB/aviation datasets (best-effort)
    "NtsbNumber","AccidentNumber","EventId","ReportStatus","EventDate","Location",
    "InjurySeverity","AircraftDamage","Make","Model","Weather","Narrative",
    "ReportNarrative","Synopsis","Analysis","ProbableCause","Findings"
]

def record_to_text(rec: Dict[str, Any]) -> str:
    """Build a compact, readable text for one record."""
    lines = []
    for k in PREFERRED_KEYS:
        if k in rec and rec[k] not in (None, "", []):
            v = rec[k]
            if isinstance(v, (dict, list)):
                try:
                    v = json.dumps(v, ensure_ascii=False)[:4000]
                except Exception:
                    v = str(v)[:4000]
            lines.append(f"{k}: {v}")
    if not lines:
        # Fallback: stringify everything (capped)
        for k, v in rec.items():
            if isinstance(v, (dict, list)):
                try:
                    v = json.dumps(v, ensure_ascii=False)[:2000]
                except Exception:
                    v = str(v)[:2000]
            else:
                v = str(v)
            lines.append(f"{k}: {v}")
    return clamp_text("\n".join(lines), max_chars=12000)

def chunk_text(text: str, chunk_chars: int = 10000) -> List[str]:
    """Split a giant raw text file into manageable chunks."""
    text = text.strip()
    if not text:
        return []
    return [text[i:i+chunk_chars] for i in range(0, len(text), chunk_chars)]

# ===== 3-step workflow for ONE record (or one chunk) =====
def analyze_one(report_text: str) -> Dict[str, str]:
    analysis = {}

    # Step 1
    m1 = [{"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=clamp_text(report_text))}]
    initial = call_llm(m1)
    analysis["initial_analysis"] = initial

    # Step 2
    m2 = [
        {"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=clamp_text(report_text))},
        {"role": "assistant", "content": initial},
        {"role": "user", "content": SELF_CRITIQUE_PROMPT.format(report_text=clamp_text(report_text), initial_analysis=initial)},
    ]
    critique = call_llm(m2)
    analysis["self_critique"] = critique

    # Step 3
    m3 = [
        {"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=clamp_text(report_text))},
        {"role": "assistant", "content": initial},
        {"role": "user", "content": SELF_CRITIQUE_PROMPT.format(report_text=clamp_text(report_text), initial_analysis=initial)},
        {"role": "assistant", "content": critique},
        {"role": "user", "content": FINAL_REVISION_PROMPT.format(
            report_text=clamp_text(report_text), initial_analysis=initial, critique_text=critique)}
    ]
    final = call_llm(m3)
    analysis["final_analysis"] = final

    return analysis

# ===== Main pipeline =====
def analyze_file(path: str, output_path: str = "analysis_result_api.txt", max_records: Optional[int] = None):
    """
    Tries to parse JSON. If it's a list, analyzes each record.
    If it's a dict, analyzes that single record.
    If not JSON, falls back to chunked raw-text analysis.
    Saves all results incrementally to the output file.
    """
    print(f"Loading from {path} ...")

    # Read raw
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Try JSON parse
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        pass

    # Prepare output
    out = open(output_path, "w", encoding="utf-8")
    def write_sep():
        out.write("\n" + "="*90 + "\n\n")

    try:
        if isinstance(data, list):
            print(f"Detected JSON array with {len(data)} records.")
            count = 0
            for rec in tqdm(data, desc="Analyzing records"):
                if max_records is not None and count >= max_records:
                    break
                try:
                    rec_text = record_to_text(rec)
                    results = analyze_one(rec_text)
                    ntsb = rec.get("NtsbNumber") or rec.get("AccidentNumber") or f"record_{count+1}"
                    out.write(f"### Record: {ntsb}\n\n")
                    out.write("--- INITIAL ANALYSIS ---\n\n" + results["initial_analysis"] + "\n\n")
                    out.write("--- SELF-CRITIQUE ---\n\n" + results["self_critique"] + "\n\n")
                    out.write("--- FINAL REVISED ANALYSIS ---\n\n" + results["final_analysis"] + "\n")
                    write_sep()
                    out.flush()  # save incrementally
                    count += 1
                except Exception as e:
                    out.write(f"[ERROR on record {count+1}]: {e}\n{traceback.format_exc()}\n")
                    write_sep()
                    out.flush()
            print(f"Done. Processed {count} record(s).")
        elif isinstance(data, dict):
            print("Detected single JSON object. Analyzing as one record.")
            rec_text = record_to_text(data)
            results = analyze_one(rec_text)
            out.write("--- INITIAL ANALYSIS ---\n\n" + results["initial_analysis"] + "\n\n")
            out.write("--- SELF-CRITIQUE ---\n\n" + results["self_critique"] + "\n\n")
            out.write("--- FINAL REVISED ANALYSIS ---\n\n" + results["final_analysis"] + "\n")
        else:
            print("Not valid JSON. Falling back to chunked raw-text analysis.")
            chunks = chunk_text(raw, chunk_chars=10000)
            for i, ch in tqdm(list(enumerate(chunks, 1)), desc="Analyzing chunks"):
                try:
                    results = analyze_one(ch)
                    out.write(f"### Chunk {i}\n\n")
                    out.write("--- INITIAL ANALYSIS ---\n\n" + results["initial_analysis"] + "\n\n")
                    out.write("--- SELF-CRITIQUE ---\n\n" + results["self_critique"] + "\n\n")
                    out.write("--- FINAL REVISED ANALYSIS ---\n\n" + results["final_analysis"] + "\n")
                    write_sep()
                    out.flush()
                except Exception as e:
                    out.write(f"[ERROR on chunk {i}]: {e}\n{traceback.format_exc()}\n")
                    write_sep()
                    out.flush()
    finally:
        out.close()
        print(f"\n✅ Results saved to {output_path}")

if __name__ == "__main__":
    FILE_PATH = r"C:\Users\elahe\Downloads\2980482a-a3e2-40b7-b486-d973b234436cAviationData.json"
    OUTPUT_PATH = "analysis_result_api.txt"

    # Tip: during testing, cap how many records to avoid long runs (set to None to process all)
    analyze_file(FILE_PATH, output_path=OUTPUT_PATH, max_records=10)
