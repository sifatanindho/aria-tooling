"""
Annotate Secret Hitler game transcripts with two schemas:
  1. Werewolf persuasion strategies (Lai et al., 2022)
  2. DeliData deliberation cues (Karadzhov et al., 2023)

Each schema is run as a SEPARATE API call so the model isn't forced to pick
between frameworks. Utterances are batched (default 8 per call) to amortize
the context-history token cost.

Usage:
    python annotate_secret_hitler.py --input game1.csv --output game1_annotated.csv
    python annotate_secret_hitler.py --input game1.csv --output game1_annotated.csv --batch-size 8 --context-window 5

CSV input requirements:
    Required columns: speaker, utterance
    Optional column:  utterance_id (auto-generated if missing)
    Other columns are preserved in the output.

Set OPENAI_API_KEY in your environment before running.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
DEFAULT_BATCH_SIZE = 8        # utterances annotated per API call
DEFAULT_CONTEXT_WINDOW = 5    # prior utterances shown as context
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 4

# Adjust these if your CSV uses different column names
SPEAKER_COL = "speaker"
UTTERANCE_COL = "text"
ID_COL = "utterance_id"

# ----------------------------------------------------------------------------
# Schema definitions
# ----------------------------------------------------------------------------

WEREWOLF_SCHEMA = {
    "name": "werewolf",
    "labels": [
        "Identity Declaration",
        "Accusation",
        "Interrogation",
        "Call for Action",
        "Defense",
        "Evidence",
        "No Strategy",
    ],
    "system_prompt": """You are annotating utterances from a Secret Hitler game using the persuasion-strategy schema from Lai et al. (2022), originally designed for One Night Ultimate Werewolf. Secret Hitler is a social-deduction game where Liberals try to identify Fascists (and Hitler) while Fascists try to deceive. The same persuasion strategies apply.

Apply these labels (multi-label — an utterance can have several):

1. Identity Declaration: State one's own role, party, or claimed identity (e.g. "I'm liberal", "i am not facist").
2. Accusation: Claim someone has a specific identity or strategic behavior (e.g. "I think Bob is a fascist", "She's clearly Hitler").
3. Interrogation: Question someone's identity or behavior (e.g. "Why did you pick him?", "What did you draw?").
4. Call for Action: Encourage the group to take an action (e.g. "Vote no on this government", "We have to investigate her next").
5. Defense: Defend oneself or someone else against an accusation, or justify a prior decision (e.g. "I picked him because he was claimed liberal", "I'm not a fascist").
6. Evidence: Provide game-related facts or information (e.g. "He passed two fascist policies last round", "The deck has 11 fascist cards").
7. No Strategy: Pleasantries, hesitation, rule clarification, or anything that doesn't fit the above.

Important rules:
- An utterance can have multiple labels. Be generous with multi-labeling when both clearly apply (e.g. "I'm liberal and Bob is a fascist" → Identity Declaration + Accusation).
- "No Strategy" is exclusive — only use it when NO other label applies.
- Reasoning takes priority over Solution-style claims when both could apply.
- Pay attention to context: "I trust him" after an accusation is Defense; the same utterance pre-vote is closer to Call for Action.

Output STRICT JSON with this structure (no prose, no markdown):
{
  "annotations": [
    {
      "utterance_id": "<id>",
      "labels": ["Accusation", "Evidence"],
      "rationale": "Brief one-sentence explanation",
      "confidence": "high"
    },
    ...
  ]
}

Confidence must be one of: "low", "medium", "high".
The text is an automated transcript of a spoken conversation, so it may contain typos, disfluencies, and informal language. Use your best judgment to interpret the intended meaning when applying the labels.
"""
}

DELIDATA_SCHEMA = {
    "name": "delidata",
    "labels": [
        # Type
        "Probing Deliberation",
        "Non-probing Deliberation",
        "None",
        # Role
        "Moderation",
        "Reasoning",
        "Solution",
        "Agree",
        "Disagree",
        # Additional
        "complete_solution",
        "partial_solution",
        "solution_summary",
        "consider_opposite",
    ],
    "system_prompt": """You are annotating utterances from a Secret Hitler game using the DeliData deliberation schema (Karadzhov et al., 2023), originally for collaborative problem-solving. Secret Hitler involves within-team collaboration (Liberals deducing together; Fascists coordinating), so this schema captures conversational dynamics that the persuasion schema misses.

Annotate each utterance on THREE dimensions:

DIMENSION 1 — Type (choose exactly one):
- "Probing Deliberation": Provokes discussion, deliberation, or argument WITHOUT introducing new information (e.g. "What do you all think?", "Why him?").
- "Non-probing Deliberation": Substantive contribution — claims, reasoning, agreement, etc.
- "None": Pleasantries, hesitation, rule discussion, off-topic.

DIMENSION 2 — Role (choose exactly one if Type is Probing or Non-probing; otherwise null):
- "Moderation" (Probing only): Manages conversation flow without engaging the task itself ("Let's hear from everyone", "Vote already").
- "Reasoning": Argumentation — why something is true or what conclusions follow (takes priority over Solution when both could apply).
- "Solution": Proposing or managing a concrete decision (vote, pick, accusation target).
- "Agree" (Non-probing only): Expresses agreement with a prior argument or proposed decision.
- "Disagree" (Non-probing only): Expresses disagreement with a prior argument or proposed decision.

DIMENSION 3 — Additional labels (zero or more, list any that apply):
- "complete_solution": Advocates for a complete decision (e.g. "Vote no and then investigate Alice").
- "partial_solution": Advocates for one part of a decision (e.g. "At least we should vote no").
- "solution_summary": Recalls earlier proposals to prompt agreement (e.g. "So we're all voting yes?").
- "consider_opposite": Suggests an opposite/alternative direction (e.g. "What if he's actually fascist?").

Note: We do NOT annotate "specific_addressee" — that requires multimodal signals.
he text is an automated transcript of a spoken conversation, so it may contain typos, disfluencies, and informal language. Use your best judgment to interpret the intended meaning when applying the labels.

Output STRICT JSON (no prose, no markdown):
{
  "annotations": [
    {
      "utterance_id": "<id>",
      "type": "Non-probing Deliberation",
      "role": "Reasoning",
      "additional": ["partial_solution"],
      "rationale": "Brief one-sentence explanation",
      "confidence": "high"
    },
    ...
  ]
}

Use null for "role" when type is "None". Use [] for "additional" when none apply. Confidence must be "low", "medium", or "high".
"""
}

# ----------------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------------

def format_context(prior_rows):
    """Format prior utterances as 'Speaker: utterance' lines."""
    if not prior_rows:
        return "(beginning of game — no prior utterances)"
    lines = [f"{r[SPEAKER_COL]}: {r[UTTERANCE_COL]}" for r in prior_rows]
    return "\n".join(lines)


def format_targets(target_rows):
    """Format target utterances with explicit IDs for the model to reference."""
    lines = []
    for r in target_rows:
        lines.append(f'[id={r[ID_COL]}] {r[SPEAKER_COL]}: {r[UTTERANCE_COL]}')
    return "\n".join(lines)


def build_user_message(context_rows, target_rows):
    return f"""CONTEXT (prior utterances, for reference only — do not annotate these):
{format_context(context_rows)}

UTTERANCES TO ANNOTATE (annotate each one, preserving the [id=...] in your output):
{format_targets(target_rows)}

Return JSON with one annotation object per target utterance, in the same order."""


# ----------------------------------------------------------------------------
# API call with retry
# ----------------------------------------------------------------------------

def call_model(client, system_prompt, user_message):
    """Call the model with retry logic. Returns parsed JSON or raises."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError as e:
            last_err = f"JSON parse error: {e}\nRaw content: {content[:500]}"
        except Exception as e:
            last_err = f"API error: {e}"

        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF_SEC * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{MAX_RETRIES} after {wait}s — {last_err}", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_err}")


# ----------------------------------------------------------------------------
# Main annotation loop
# ----------------------------------------------------------------------------

def annotate_dataframe(df, client, schema, batch_size, context_window):
    """Annotate the full dataframe with one schema. Returns dict: id -> annotation."""
    n = len(df)
    annotations = {}
    rows = df.to_dict("records")

    print(f"\n[{schema['name']}] Annotating {n} utterances in batches of {batch_size}...")

    i = 0
    while i < n:
        batch_end = min(i + batch_size, n)
        target_rows = rows[i:batch_end]

        # Context: window-many utterances before the FIRST target in this batch
        ctx_start = max(0, i - context_window)
        context_rows = rows[ctx_start:i]

        user_msg = build_user_message(context_rows, target_rows)

        try:
            result = call_model(client, schema["system_prompt"], user_msg)
            batch_annotations = result.get("annotations", [])

            # Map by id; warn on missing or extra
            returned_ids = set()
            for ann in batch_annotations:
                uid = str(ann.get("utterance_id", "")).strip()
                if uid:
                    annotations[uid] = ann
                    returned_ids.add(uid)

            expected_ids = {str(r[ID_COL]) for r in target_rows}
            missing = expected_ids - returned_ids
            if missing:
                print(f"  WARNING: batch {i}-{batch_end} missing annotations for ids: {missing}", file=sys.stderr)

        except Exception as e:
            print(f"  ERROR on batch {i}-{batch_end}: {e}", file=sys.stderr)
            # Fill with empty annotations so downstream doesn't break
            for r in target_rows:
                annotations[str(r[ID_COL])] = {"utterance_id": r[ID_COL], "error": str(e)}

        # Progress indicator
        pct = 100 * batch_end / n
        print(f"  [{schema['name']}] {batch_end}/{n} ({pct:.0f}%)")

        i = batch_end

    return annotations


# ----------------------------------------------------------------------------
# Output formatting
# ----------------------------------------------------------------------------

def merge_annotations(df, werewolf_annotations, delidata_annotations):
    """Add annotation columns to the dataframe."""
    out = df.copy()

    # Werewolf columns
    out["ww_labels"] = out[ID_COL].astype(str).map(
        lambda uid: ";".join(werewolf_annotations.get(uid, {}).get("labels", []))
    )
    out["ww_rationale"] = out[ID_COL].astype(str).map(
        lambda uid: werewolf_annotations.get(uid, {}).get("rationale", "")
    )
    out["ww_confidence"] = out[ID_COL].astype(str).map(
        lambda uid: werewolf_annotations.get(uid, {}).get("confidence", "")
    )

    # DeliData columns
    out["dd_type"] = out[ID_COL].astype(str).map(
        lambda uid: delidata_annotations.get(uid, {}).get("type", "")
    )
    out["dd_role"] = out[ID_COL].astype(str).map(
        lambda uid: delidata_annotations.get(uid, {}).get("role") or ""
    )
    out["dd_additional"] = out[ID_COL].astype(str).map(
        lambda uid: ";".join(delidata_annotations.get(uid, {}).get("additional", []) or [])
    )
    out["dd_rationale"] = out[ID_COL].astype(str).map(
        lambda uid: delidata_annotations.get(uid, {}).get("rationale", "")
    )
    out["dd_confidence"] = out[ID_COL].astype(str).map(
        lambda uid: delidata_annotations.get(uid, {}).get("confidence", "")
    )

    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Annotate Secret Hitler transcripts.")
    parser.add_argument("--input", required=True, help="Path to input CSV/TSV.")
    parser.add_argument("--output", required=True, help="Path to output CSV.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Utterances per API call (default {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW,
                        help=f"Prior utterances as context (default {DEFAULT_CONTEXT_WINDOW}).")
    parser.add_argument("--schema", choices=["werewolf", "delidata", "both"], default="both",
                        help="Which schema(s) to run.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Annotate only the first N utterances (for testing).")
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save raw annotations as JSON files.")
    args = parser.parse_args()

    # Load
    sep = "\t" if args.input.endswith(".tsv") else ","
    df = pd.read_csv(args.input, sep=sep)
    print(f"Loaded {len(df)} utterances from {args.input}")
    print(f"Columns: {list(df.columns)}")

    # Validate required columns
    if SPEAKER_COL not in df.columns or UTTERANCE_COL not in df.columns:
        sys.exit(f"ERROR: CSV must contain '{SPEAKER_COL}' and '{UTTERANCE_COL}' columns. "
                 f"Edit the constants at the top of the script if your columns are named differently.")

    # Auto-generate IDs if missing
    if ID_COL not in df.columns:
        df[ID_COL] = [f"u{i:04d}" for i in range(len(df))]
        print(f"Auto-generated '{ID_COL}' column.")

    # Drop empty utterances
    before = len(df)
    df = df[df[UTTERANCE_COL].notna() & (df[UTTERANCE_COL].astype(str).str.strip() != "")].reset_index(drop=True)
    if len(df) < before:
        print(f"Dropped {before - len(df)} empty utterances.")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {len(df)} utterances.")

    # API client
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set in environment.")
    client = OpenAI()

    # Run
    werewolf_annotations = {}
    delidata_annotations = {}

    if args.schema in ("werewolf", "both"):
        werewolf_annotations = annotate_dataframe(
            df, client, WEREWOLF_SCHEMA, args.batch_size, args.context_window
        )

    if args.schema in ("delidata", "both"):
        delidata_annotations = annotate_dataframe(
            df, client, DELIDATA_SCHEMA, args.batch_size, args.context_window
        )

    # Merge and save
    out_df = merge_annotations(df, werewolf_annotations, delidata_annotations)
    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved annotated CSV to {out_path}")

    if args.save_raw:
        if werewolf_annotations:
            raw_path = out_path.with_suffix(".werewolf.json")
            with open(raw_path, "w") as f:
                json.dump(werewolf_annotations, f, indent=2)
            print(f"Saved raw werewolf annotations to {raw_path}")
        if delidata_annotations:
            raw_path = out_path.with_suffix(".delidata.json")
            with open(raw_path, "w") as f:
                json.dump(delidata_annotations, f, indent=2)
            print(f"Saved raw delidata annotations to {raw_path}")

    # Quick summary
    print("\n=== Summary ===")
    if werewolf_annotations:
        all_labels = []
        for ann in werewolf_annotations.values():
            all_labels.extend(ann.get("labels", []))
        print(f"Werewolf label counts: {pd.Series(all_labels).value_counts().to_dict()}")
    if delidata_annotations:
        types = [ann.get("type", "") for ann in delidata_annotations.values()]
        print(f"DeliData type counts: {pd.Series(types).value_counts().to_dict()}")


if __name__ == "__main__":
    main()  