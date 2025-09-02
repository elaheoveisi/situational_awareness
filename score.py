# score.py
# ---------------------------------------------
# LLM7 (OpenAI-compatible) accident analysis pipeline
# - Processes JSON arrays per record (or raw text in chunks)
# - 3-step analysis (Initial -> Critique -> Final)
# - Self-Verification Score (SVS) per record
# - Retries/backoff + prompt size clamping
# - Per-record CSV export + dataset summary
#
# Usage:
#   pip install --upgrade openai tqdm
#   python score.py
# ---------------------------------------------

import json
import time
import traceback
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# --- OpenAI-compatible client (llm7) ---
# pip install --upgrade openai
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

LLM7_BASE_URL = "https://api.llm7.io/v1"
LLM7_MODEL = "gpt-4o-mini-2024-07-18"

# llm7 accepts any string; no real key needed
client = OpenAI(base_url=LLM7_BASE_URL, api_key="unused")

# ===== Prompts =====
INITIAL_ANALYSIS_PROMPT = """
You are an expert aviation accident analyst. Analyze THIS ONE accident record and determine whether situational awareness (SA) loss was a contributing factor. Situational awareness is knowing what’s going on around you so you can decide what to do next: first you notice key cues (perception), then you understand what they mean (comprehension), and finally you anticipate what will happen if nothing changes (projection). For example, in a drone-based search-and-rescue mission, the pilot perceives low battery, rising wind, and a heat signature on the video feed; the rescuer comprehends that the heat signature likely indicates a person near a tree line and that the wind will push the drone off course; together they project that, without action, the drone may lose power before reaching the target. Based on that awareness, they decide to fly a shorter route, drop altitude to shield from wind, and call in a ground team to converge on the coordinates. If any part fails—missed cues, wrong interpretation, or poor prediction—performance drops (e.g., the drone diverts too late or the team arrives in the wrong place). That continuous loop of perceive–understand–predict is situational awareness.

Your answer must have exactly two parts:
1) Conclusion: "Yes" or "No".
2) Reasoning: step-by-step, citing details FROM THE TEXT below. If "Yes", name likely SA-loss drivers (e.g., distraction, channelized attention, overload, fatigue, weather).
3) Write a list of the reason of the SA loss (each no more than 3 words).

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
2) Alternatives: Did it miss other plausible causes/mechanisms stated/implied in the record?
3) Conclusion Strength: Is the Yes/No warranted by the evidence?
4) Bias: Any assumptions beyond the text?

Provide a structured critique.
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
def clamp_text(text: str, max_chars: int = 12000) -> str:
    """Truncate text to avoid overlong payloads."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    note = f"\n\n[TRUNCATED: original length {len(text)} chars]"
    return head + note

PREFERRED_KEYS = [
    # Common best-effort fields
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

def _extract_json(s: str) -> dict:
    """
    Robustly extract the first JSON object from a model reply.
    Handles code fences and extra commentary.
    """
    s = s.strip()

    # 1) Prefer fenced blocks ```json ... ``` or ``` ... ```
    fence = "```"
    if fence in s:
        parts = s.split(fence)
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            # strip language tag if present
            first_line = block.split("\n", 1)[0].lower().strip()
            if first_line in ("json", "javascript"):
                block = block.split("\n", 1)[1] if "\n" in block else ""
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
                obj, end = dec.raw_decode(s[start:])
                return obj
            except Exception:
                continue

    # 3) Fallback: non-greedy regex for smallest {...}
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
        except (APITimeoutError, APIConnectionError, RateLimitError):
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status in {500, 502, 503, 504}:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                raise
        except Exception:
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    raise RuntimeError("LLM request failed after retries.")

def score_self_verification(report_text: str, final_analysis: str) -> dict:
    """Return dict with subscores, overall_svs, and notes."""
    prompt = SELF_VERIFICATION_PROMPT_JSON.format(
        report_text=clamp_text(report_text, 9000),
        final_analysis=clamp_text(final_analysis, 5000),
    )
    msg = [{"role": "user", "content": prompt}]
    try:
        raw = call_llm(msg, temperature=0.0, max_tokens=600)  # lower variance for scoring
        data = _extract_json(raw)
    except Exception:
        # Fallback: try again with a stricter JSON-only reminder
        strict = prompt + "\n\nReturn ONLY the JSON. Do not add any commentary."
        raw = call_llm([{"role": "user", "content": strict}], temperature=0.0, max_tokens=600)
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

    # Step 1
    m1 = [{"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=clamp_text(report_text))}]
    initial = call_llm(m1)
    analysis["initial_analysis"] = initial

    # Step 2
    m2 = [
        {"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=clamp_text(report_text))},
        {"role": "assistant", "content": initial},
        {"role": "user", "content": SELF_CRITIQUE_PROMPT.format(
            report_text=clamp_text(report_text), initial_analysis=initial)},
    ]
    critique = call_llm(m2)
    analysis["self_critique"] = critique

    # Step 3
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
def _append_svs_csv_row(row_id: str, subs: dict, overall: Any, csv_path: str = "svs_per_record.csv"):
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
def analyze_file(path: str, output_path: str = "analysis_result_api.txt", max_records: Optional[int] = None):
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

                    # aggregate + CSV (NO stray try:)
                    if sv:
                        agg["count"] += 1
                        for key in ("evidence_grounding","alt_explanations","conclusion_strength","bias_control","critique_integration"):
                            agg[key] += float(subs.get(key, 0) or 0)
                        agg["overall_svs"] += float(sv.get("overall_svs", 0) or 0)
                        # CSV is best-effort
                        try:
                            _append_svs_csv_row(ntsb, subs, sv.get("overall_svs", ""))
                        except Exception:
                            pass

                    write_sep()
                    out.flush()
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
                    _append_svs_csv_row(row_id, subs, sv.get("overall_svs", ""))
                except Exception:
                    pass

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
                            _append_svs_csv_row(f"chunk_{i}", subs, sv.get("overall_svs", ""))
                        except Exception:
                            pass

                    write_sep()
                    out.flush()
                except Exception as e:
                    out.write(f"[ERROR on chunk {i}]: {e}\n{traceback.format_exc()}\n")
                    write_sep()
                    out.flush()
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
    OUTPUT_PATH = "analysis_result_api.txt"

    # Tip: during testing, set a number like 10; set to None to process all records
    MAX_RECORDS = None

    analyze_file(FILE_PATH, output_path=OUTPUT_PATH, max_records=MAX_RECORDS)
