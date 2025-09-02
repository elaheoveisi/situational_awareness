import os
import json
from tqdm import tqdm
import time

# If you really need pandas/torch/transformers later, keep them.
# For this script they're not required:
# import pandas as pd
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# === 0) OpenAI-compatible client pointing at llm7 ===
# pip install --upgrade openai
from openai import OpenAI

LLM7_BASE_URL = "https://api.llm7.io/v1"
LLM7_MODEL = "gpt-4o-mini-2024-07-18"

client = OpenAI(
    base_url=LLM7_BASE_URL,
    api_key="unused"  # per llm7, no key needed
)

# --- 1) LLM call using Chat Completions (replaces local HF pipeline) ---
def call_llm(messages, *, temperature=0.3, max_tokens=1500):
    """
    messages: list of {"role": "...", "content": "..."} dicts
    returns: str (assistant content)
    """
    resp = client.chat.completions.create(
        model=LLM7_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# --- 2) Your prompts (keep your full versions in your file) ---
INITIAL_ANALYSIS_PROMPT = """
You are an expert aviation accident analyst. Your task is to determine if a loss of Situational Awareness (SA) was a primary contributing factor in the following accident report. Read the report carefully and regard their NtsbNumber.

Situational awareness is the ability to know what is going on around you. It involves three key steps: **perceiving** critical elements in the environment, **comprehending** their meaning, and **projecting** what will happen next. Mastering this skill is crucial for effective decision-making in complex situations, allowing you to stay ahead of events rather than just reacting to them.

For instance, during a nighttime approach in bad weather, a pilot might perceive a sudden bang, a sharp yaw to the left, and multiple warning lights. Instead of seeing these as separate issues, the pilot comprehends that they signify a catastrophic engine fire. From this understanding, she projects that without immediate action, the aircraft will struggle to reach the runway and the fire could spread to the wing. This complete picture—moving from raw data to a predictive understanding—is what allows her to execute the correct emergency procedures and land the aircraft safely.

you should look at each accident according to their NtsbNumber and answer the questions.

Your analysis must contain two parts:
1. **Conclusion**: A clear "Yes" or "No" on whether SA loss was a factor.
2. **Reasoning**: A step-by-step explanation for your conclusion, citing specific details from the report. If yes, identify the likely cause of the SA loss (e.g., distraction, channelized attention, information overload, complacency, fatigue, weather, etc).

Here is the report:
---
{report_text}
---
"""

SELF_CRITIQUE_PROMPT = """
You are a quality assurance specialist reviewing an analysis of an accident report. Your task is to critique the following analysis based ONLY on the provided report text.

**Report Text:**
---
{report_text}
---

**Initial Analysis to Critique:**
---
{initial_analysis}
---

Critique the analysis using this checklist. Be strict and objective:
1. **Evidence Check**: Is every claim in the reasoning directly supported by specific evidence from the report? Point out any unsupported claims.
2. **Alternative Explanation Check**: Did the analysis overlook other plausible explanations for the pilot's actions (e.g., mechanical failure, extreme weather, physiological issues mentioned in the report)?
3. **Conclusion Strength**: Does the evidence strongly support the "Yes/No" conclusion, or is the connection weak?
4. **Bias Check**: Does the analysis make assumptions not explicitly stated in the text?

Provide your critique in a structured format based on the checklist above.
"""

FINAL_REVISION_PROMPT = """
You are the original aviation analyst. You will revise your initial analysis using the critique. Keep the same two-part structure (Conclusion, Reasoning). Ensure every claim is grounded in the report text, address alternative explanations raised by the critique, and strengthen or temper your conclusion appropriately.

**Report Text:**
---
{report_text}
---

**Your Initial Analysis:**
---
{initial_analysis}
---

**Critique You Must Address:**
---
{critique_text}
---

Now provide the final, improved analysis.
"""

# --- 3) Main 3-step workflow (unchanged except it uses call_llm -> llm7) ---
def analyze_report(report_text):
    """
    Runs a single report through the full 3-step analysis workflow.
    """
    analysis_results = {}

    print("\nStep 1: Generating Initial Analysis (via llm7)...")
    messages_step1 = [
        {"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=report_text)}
    ]
    initial_analysis = call_llm(messages_step1)
    analysis_results['initial_analysis'] = initial_analysis

    print("Step 2: Performing Self-Critique (via llm7)...")
    messages_step2 = [
        {"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=report_text)},
        {"role": "assistant", "content": initial_analysis},
        {"role": "user", "content": SELF_CRITIQUE_PROMPT.format(
            report_text=report_text, initial_analysis=initial_analysis)}
    ]
    critique = call_llm(messages_step2)
    analysis_results['self_critique'] = critique

    print("Step 3: Generating Final Revision (via llm7)...")
    messages_step3 = [
        {"role": "user", "content": INITIAL_ANALYSIS_PROMPT.format(report_text=report_text)},
        {"role": "assistant", "content": initial_analysis},
        {"role": "user", "content": SELF_CRITIQUE_PROMPT.format(
            report_text=report_text, initial_analysis=initial_analysis)},
        {"role": "assistant", "content": critique},
        {"role": "user", "content": FINAL_REVISION_PROMPT.format(
            report_text=report_text, initial_analysis=initial_analysis, critique_text=critique)}
    ]
    final_analysis = call_llm(messages_step3)
    analysis_results['final_analysis'] = final_analysis

    return analysis_results

# --- 4) Main Execution ---
if __name__ == "__main__":
    # DEFINE THE PATH TO YOUR FILE HERE
    file_path = r"C:\Users\elahe\Downloads\2980482a-a3e2-40b7-b486-d973b234436cAviationData.json"

    print(f"Loading the entire file as one report from {file_path}...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            report_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        raise SystemExit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise SystemExit(1)

    print("Starting analysis of the report...")
    final_result = analyze_report(report_text)

    # --- Output Results to a Text File ---
    output_filename = 'analysis_result_api.txt'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("--- INITIAL ANALYSIS ---\n\n")
        f.write(final_result['initial_analysis'])
        f.write("\n\n--- SELF-CRITIQUE ---\n\n")
        f.write(final_result['self_critique'])
        f.write("\n\n--- FINAL REVISED ANALYSIS ---\n\n")
        f.write(final_result['final_analysis'])

    print(f"\n✅ Analysis complete. Results saved to {output_filename}")
