# gamma.py
# ---------------------------------------------
# Local (Ollama) accident analysis pipeline
# - Processes JSON arrays per record (or raw text in chunks)
# - 3-step analysis (Initial -> Critique -> Final)
# - Self-Verification Score (SVS) per record
# - Retries/backoff + longer timeouts for local models
# - Per-record CSV export + dataset summary
#
# Usage:
#   pip install --upgrade openai tqdm
#   (make sure Ollama is running and you pulled a model, e.g.:
#       ollama pull gemma:2b-instruct
#       ollama run gemma:2b-instruct "hi")
#   python gamma.py
# ---------------------------------------------

import json
import time
import traceback
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# --- OpenAI-compatible client (Ollama local) ---
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

# ===== Model / Server config =====
LLM7_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible server
LLM7_MODEL =  "gemma:2b"
           # Start small for speed; can try "gemma2:9b" later

# No real key needed for local servers that emulate OpenAI API
client = OpenAI(base_url=LLM7_BASE_URL, api_key="ollama")

# Optional quick warm-up (ignore errors)
try:
    _ = client.chat.completions.create(
        model=LLM7_MODEL,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=4,
        timeout=60,
        extra_body={"options": {"num_ctx": 2048}}
    )
except Exception:
    pass

# ===== Prompts =====
INITIAL_ANALYSIS_PROMPT = """
You are an expert aviation accident analyst. Analyze THIS ONE accident record and determine whether situational awareness (SA) loss was a contributing factor. Situational awareness is knowing what’s going on so you can decide what to do next: (1) perception of key cues, (2) comprehension of meaning, (3) projection of what will happen if nothing changes.

Your answer must have exactly two labeled parts:
1) Conclusion: "Yes" or "No".
2) Reasoning: brief, step-by-step, citing details FROM THE TEXT below. If "Yes", name likely SA-loss drivers (e.g., distraction, channelized attention, overload, fatigue, weather).
3) List: short bullet list of SA-loss reasons (each ≤3 words).

--- ACCIDENT RECORD ---
{report_text}
"""

SELF_CRITIQUE_PROMPT = """
You are a QA reviewer. Critique the analysis strictly using ONLY the given record text.

--- RECORD TEXT ---
{report_text}

--- INITIAL ANALYSIS ---
{initial_analysis}

Checklist:
1) Evidence Check: Are all claims supported by the record? Flag unsupported ones.
2) Alternatives: Note plausible causes/mechanisms stated or implied in the record that were missed.
3) Conclusion Strength: Is the Yes/No warranted?
4) Bias: Any assumptions beyond the text?

Provide a concise, structured critique.
"""

FINAL_REVISION_PROMPT = """
Revise the initial analysis using the critique. Keep the two-part structure:
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

# ---------- Self-Verification (SVS) ----------
# IMPORTANT: Double braces {{ }} for literal JSON braces in a .format string.
SELF_VERIFICATION_PROMPT_JSON = """
You are auditing an aviation analysis for self-verification. Use ONLY the Record Text to judge the Final Analysis.
Score each criterion from 0 to 100 (higher is better). Be strict.

CRITERIA:
1) evidence_grounding: Claims in the Final Analysis are explicitly supported by the Record Text.
2) alt_explanations: The Final Analysis considers or rules out plausible alternatives mentioned/implied in the Record Text.
3) conclusion_strength: The Yes/No conclusion logically follows from evidence; neither over- nor under-stated.
4) bias_control: The Final Analysis avoids assumptions not in the Record Text; language is precise and non-speculative.
5) critique_integration: The Final Analysis addresses the critique’s key points accurately.

Return ONLY valid JSON with this exact schema:
{{
  "scores": {{
    "evidence_grounding": <int 0-100>,
    "alt_explanations": <int 0-100>,
    "conclusion_strength": <int 0-100>,
    "bias_control": <int 0-100>,
    "critique_integration": <int 0-100>
  }},
  "notes": "1-3 short bullet notes with concrete references to the Record Text (no new facts)."
}}

Record Text:
---
{report_text}
---

Final Analysis:
---
{final_analysis}
---
"""

# ===== Utilities =====
def clamp_text(text: str, max_chars: int = 3000) -> str:
    """Truncate text to reduce load on small local models."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    note = f"\n\n[TRUNCATED: original length {len(text)} chars]"
    return head + note

PREFERRED_KEYS = [
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
                    v = json.dumps(v, ensure_ascii=False)[:2000]
                except Exception:
                    v = str(v)[:2000]
            lines.append(f"{k}: {v}")
    if not lines:
        for k, v in rec.items():
            if isinstance(v, (dict, list)):
                try:
                    v = json.dumps(v, ensure_ascii=False)[:1500]
                except Exception:
                    v = str(v)[:1500]
            else:
                v = str(v)
            lines.append(f"{k}: {v}")
    return clamp_text("\n".join(lines), max_chars=3000)

def chunk_text(text: str, chunk_chars: int = 3000) -> List[str]:
    """Split a giant raw text file into manageable chunks."""
    text = (text or "").strip()
    if not text:
        return []
    return [text[i:i+chunk_chars] for i in range(0, len(text), chunk_chars)]

def _extract_json(s: str) -> dict:
    """
    Robustly extract the first JSON object from a model reply.
    Handles code fences and extra commentary.
    """
    s = (s or "").strip()

    # 1) Prefer fenced blocks ```json ... ``` or ``` ... ```
    fence = "```"
    if fence in s:
        parts = s.split(fence)
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            # strip language tag if present
            if "\n" in block:
                first_line, rest = block.split("\n", 1)
            else:
                first_line, rest = block, ""
            if first_line.lower().strip() in ("json", "javascript"):
                block = rest
            try:
                return json.loads(block.strip())
            except Exception:
                pass  # try the next fenced block

    # 2) Try JSONDecoder from each '{'
    first_lc = s.find("{")
    if first_lc != -1:
        dec = json.JSONDecoder()
        for start in range(first_lc, len(s)):
            if s[start] != "{":
                continue
            try:
                obj, _ = dec.raw_decode(s[start:])
                return obj
            except Exception:
                continue

    # 3) Fallback: non-greedy smallest {...}
    m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    raise ValueError("No valid JSON object found in model response.")

def compute_overall_svs(subscores: dict) -> float:
    # weights sum to 1.0
    w = {
        "evidence_grounding":   0.30,
        "alt_explanations":     0.20,
        "conclusion_strength":  0.25,
        "bias_control":         0.15,
        "critique_integration": 0.10,
    }
    total = 0.0
    for k, wk in w.items():
        v = float(subscores.get(k, 0) or 0)
        total += wk * max(0.0, min(100.0, v))
    return round(total, 1)

# ===== LLM call with retries/backoff =====
def call_llm(messages: List[Dict[str, str]],
             temperature: float = 0.2,
             max_tokens: int = 350,
             max_retries: int = 6,
             base_delay: float = 1.0) -> str:
    """
    Robust call with retries on transient/server errors.
    Tuned for local models (longer timeouts).
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LLM7_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=300,  # local models may need more time
                extra_body={"options": {"num_ctx": 2048}}  # Ollama hint (if supported)
            )
            return resp.choices[0].message.content.strip()
        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            last_err = e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
        except APIStatusError as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status in {500, 502, 503, 504}:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            last_err = e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    raise RuntimeError(f"LLM request failed after retries. Last error: {last_err}")

def score_self_verification(report_text: str, final_analysis: str) -> dict:
    """Return dict with subscores, overall_svs, and notes."""
    prompt = SELF_VERIFICATION_PROMPT_JSON.format(
        report_text=clamp_text(report_text, 5000),
        final_analysis=clamp_text(final_analysis, 1800),
    )
    msg = [{"role": "user", "content": prompt}]
    try:
        raw = call_llm(msg, temperature=0.0, max_tokens=300)  # keep short
        data = _extract_json(raw)
    except Exception:
        # Fallback: try again with a stricter JSON-only reminder
        strict = prompt + "\n\nReturn ONLY the JSON. Do not add any commentary."
        raw = call_llm([{"role": "user", "content": strict}], temperature=0.0, max_tokens=250)
        data = _extract_json(raw)

    subs = data.get("scores", {})
    overall = compute_overall_svs(subs)
    return {
        "scores": subs,
        "overall_svs": overall,
        "notes": data.get("notes", "")
    }

# ===== 3-step workflow for ONE record (or one chunk) =====
def analyze_one(report_text: str) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {}

    # Step 1: Initial analysis
    m1 = [{"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=clamp_text(report_text))}]
    initial = call_llm(m1)
    analysis["initial_analysis"] = initial

    # Step 2: Self-critique
    m2 = [
        {"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=clamp_text(report_text))},
        {"role": "assistant", "content": initial},
        {"role": "user", "content": SELF_CRITIQUE_PROMPT.format(
            report_text=clamp_text(report_text), initial_analysis=initial)},
    ]
    critique = call_llm(m2)
    analysis["self_critique"] = critique

    # Step 3: Final revision
    m3 = [
        {"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=clamp_text(report_text))},
        {"role": "assistant", "content": initial},
        {"role": "user", "content": SELF_CRITIQUE_PROMPT.format(
            report_text=clamp_text(report_text), initial_analysis=initial)},
        {"role": "assistant", "content": critique},
        {"role": "user", "content": FINAL_REVISION_PROMPT.format(
            report_text=clamp_text(report_text), initial_analysis=initial, critique_text=critique)}
    ]
    final = call_llm(m3)
    analysis["final_analysis"] = final

    # Self-Verification Score
    sv = score_self_verification(report_text, final)
    analysis["self_verification"] = sv

    return analysis

# ===== CSV helper =====
def _append_svs_csv_row(row_id: str, subs: dict, overall: Any, csv_path: str):
    """Append one SVS row to CSV (creates file with header if needed)."""
    try:
        from pathlib import Path
        p = Path(csv_path)
        new_file = not p.exists() or p.stat().st_size == 0
        with p.open("a", encoding="utf-8") as csvf:
            if new_file:
                csvf.write("id,evidence_grounding,alt_explanations,conclusion_strength,bias_control,critique_integration,overall_svs\n")
            csvf.write(
                f"{row_id},"
                f"{subs.get('evidence_grounding', '')},"
                f"{subs.get('alt_explanations', '')},"
                f"{subs.get('conclusion_strength', '')},"
                f"{subs.get('bias_control', '')},"
                f"{subs.get('critique_integration', '')},"
                f"{overall}\n"
            )
    except Exception:
        # CSV is best-effort; don't crash the pipeline for it
        pass

# ===== Main pipeline =====
def analyze_file(path: str, output_path: str, csv_path: str, max_records: Optional[int] = None):
    """
    Tries to parse JSON. If it's a list, analyzes each record.
    If it's a dict, analyzes that single record.
    If not JSON, falls back to chunked raw-text analysis.
    Saves all results incrementally to the output file and prints an SVS summary.
    """
    print(f"Loading from {path} ...")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Try JSON parse
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        pass

    # Prepare output and aggregator
    out = open(output_path, "w", encoding="utf-8")

    def write_sep():
        out.write("\n" + "="*90 + "\n\n")

    agg = {
        "count": 0,
        "evidence_grounding": 0.0,
        "alt_explanations": 0.0,
        "conclusion_strength": 0.0,
        "bias_control": 0.0,
        "critique_integration": 0.0,
        "overall_svs": 0.0,
    }

    try:
        if isinstance(data, list):
            print(f"Detected JSON array with {len(data)} records.")
            count = 0
            try:
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

                        sv = results.get("self_verification", {})
                        subs = sv.get("scores", {})
                        out.write("\n--- SELF-VERIFICATION SCORE ---\n")
                        out.write(f"Overall SVS: {sv.get('overall_svs', 'NA')}\n")
                        out.write(f"  - evidence_grounding:    {subs.get('evidence_grounding', 'NA')}\n")
                        out.write(f"  - alt_explanations:      {subs.get('alt_explanations', 'NA')}\n")
                        out.write(f"  - conclusion_strength:   {subs.get('conclusion_strength', 'NA')}\n")
                        out.write(f"  - bias_control:          {subs.get('bias_control', 'NA')}\n")
                        out.write(f"  - critique_integration:  {subs.get('critique_integration', 'NA')}\n")
                        notes = sv.get("notes", "")
                        if notes:
                            out.write(f"Notes: {notes}\n")

                        # aggregate + CSV (best effort)
                        if sv:
                            agg["count"] += 1
                            for key in ("evidence_grounding","alt_explanations","conclusion_strength","bias_control","critique_integration"):
                                agg[key] += float(subs.get(key, 0) or 0)
                            agg["overall_svs"] += float(sv.get("overall_svs", 0) or 0)
                            try:
                                _append_svs_csv_row(ntsb, subs, sv.get("overall_svs", ""), csv_path)
                            except Exception:
                                pass

                        write_sep()
                        out.flush()
                        count += 1
                    except Exception as e:
                        out.write(f"[ERROR on record {count+1}]: {e}\n{traceback.format_exc()}\n")
                        write_sep()
                        out.flush()
            except KeyboardInterrupt:
                print("\n⏹️ Stopped by user. Writing partial results...")

            print(f"Done. Processed {count} record(s).")

        elif isinstance(data, dict):
            print("Detected single JSON object. Analyzing as one record.")
            rec_text = record_to_text(data)
            results = analyze_one(rec_text)
            out.write("--- INITIAL ANALYSIS ---\n\n" + results["initial_analysis"] + "\n\n")
            out.write("--- SELF-CRITIQUE ---\n\n" + results["self_critique"] + "\n\n")
            out.write("--- FINAL REVISED ANALYSIS ---\n\n" + results["final_analysis"] + "\n")

            sv = results.get("self_verification", {})
            subs = sv.get("scores", {})
            out.write("\n--- SELF-VERIFICATION SCORE ---\n")
            out.write(f"Overall SVS: {sv.get('overall_svs', 'NA')}\n")
            out.write(f"  - evidence_grounding:    {subs.get('evidence_grounding', 'NA')}\n")
            out.write(f"  - alt_explanations:      {subs.get('alt_explanations', 'NA')}\n")
            out.write(f"  - conclusion_strength:   {subs.get('conclusion_strength', 'NA')}\n")
            out.write(f"  - bias_control:          {subs.get('bias_control', 'NA')}\n")
            out.write(f"  - critique_integration:  {subs.get('critique_integration', 'NA')}\n")
            notes = sv.get("notes", "")
            if notes:
                out.write(f"Notes: {notes}\n")

            if sv:
                agg["count"] += 1
                for key in ("evidence_grounding","alt_explanations","conclusion_strength","bias_control","critique_integration"):
                    agg[key] += float(subs.get(key, 0) or 0)
                agg["overall_svs"] += float(sv.get("overall_svs", 0) or 0)
                try:
                    row_id = data.get("NtsbNumber") or data.get("AccidentNumber") or "single_record"
                    _append_svs_csv_row(row_id, subs, sv.get("overall_svs", ""), csv_path)
                except Exception:
                    pass

        else:
            print("Not valid JSON. Falling back to chunked raw-text analysis.")
            chunks = chunk_text(raw, chunk_chars=3000)
            try:
                for i, ch in tqdm(list(enumerate(chunks, 1)), desc="Analyzing chunks"):
                    try:
                        results = analyze_one(ch)
                        out.write(f"### Chunk {i}\n\n")
                        out.write("--- INITIAL ANALYSIS ---\n\n" + results["initial_analysis"] + "\n\n")
                        out.write("--- SELF-CRITIQUE ---\n\n" + results["self_critique"] + "\n\n")
                        out.write("--- FINAL REVISED ANALYSIS ---\n\n" + results["final_analysis"] + "\n")

                        sv = results.get("self_verification", {})
                        subs = sv.get("scores", {})
                        out.write("\n--- SELF-VERIFICATION SCORE ---\n")
                        out.write(f"Overall SVS: {sv.get('overall_svs', 'NA')}\n")
                        out.write(f"  - evidence_grounding:    {subs.get('evidence_grounding', 'NA')}\n")
                        out.write(f"  - alt_explanations:      {subs.get('alt_explanations', 'NA')}\n")
                        out.write(f"  - conclusion_strength:   {subs.get('conclusion_strength', 'NA')}\n")
                        out.write(f"  - bias_control:          {subs.get('bias_control', 'NA')}\n")
                        out.write(f"  - critique_integration:  {subs.get('critique_integration', 'NA')}\n")
                        notes = sv.get("notes", "")
                        if notes:
                            out.write(f"Notes: {notes}\n")

                        if sv:
                            agg["count"] += 1
                            for key in ("evidence_grounding","alt_explanations","conclusion_strength","bias_control","critique_integration"):
                                agg[key] += float(subs.get(key, 0) or 0)
                            agg["overall_svs"] += float(sv.get("overall_svs", 0) or 0)
                            try:
                                _append_svs_csv_row(f"chunk_{i}", subs, sv.get("overall_svs", ""), csv_path)
                            except Exception:
                                pass

                        write_sep()
                        out.flush()
                    except Exception as e:
                        out.write(f"[ERROR on chunk {i}]: {e}\n{traceback.format_exc()}\n")
                        write_sep()
                        out.flush()
            except KeyboardInterrupt:
                print("\n⏹️ Stopped by user. Writing partial results...")

    finally:
        # Dataset-level SVS summary
        if agg["count"] > 0:
            n = agg["count"]
            summary = [
                "\n=== SVS Summary ===",
                f"Records scored: {n}",
                f"evidence_grounding:   {round(agg['evidence_grounding']/n, 1)}",
                f"alt_explanations:     {round(agg['alt_explanations']/n, 1)}",
                f"conclusion_strength:  {round(agg['conclusion_strength']/n, 1)}",
                f"bias_control:         {round(agg['bias_control']/n, 1)}",
                f"critique_integration: {round(agg['critique_integration']/n, 1)}",
                f"Overall SVS:          {round(agg['overall_svs']/n, 1)}",
                ""
            ]
            print("\n".join(summary))
            out.write("\n".join(summary) + "\n")
        out.close()
        print(f"\n✅ Results saved to {output_path}")

# ===== Entry Point =====
if __name__ == "__main__":
    # Set your input file here:
    FILE_PATH = r"C:\Users\elahe\Downloads\2980482a-a3e2-40b7-b486-d973b234436cAviationData.json"

    # Output text file (per-run log)
    OUTPUT_PATH = "analysis_result_gamma.txt"

    # CSV path for per-record SVS rows
    CSV_PATH = "svsgamma.csv"  # e.g., r"C:\Users\elahe\Documents\svsgamma.csv"

    # Tip: during testing, process a few records first
    MAX_RECORDS = None  # set to None to process all

    analyze_file(FILE_PATH, output_path=OUTPUT_PATH, csv_path=CSV_PATH, max_records=MAX_RECORDS)
