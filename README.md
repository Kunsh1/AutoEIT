# AutoEIT — Automated EIT Transcription Pipeline
### GSoC 2026 Evaluation | HumanAI Foundation (NIU + University of Alabama)

---

## Results

| Participant | File | Responses Filled | Notes |
|:---:|---|:---:|---|
| 038010 | EIT-2A | 29/30 | Sentence 19: no detectable response |
| 038011 | EIT-1A | 30/30 | |
| 038012 | EIT-2A | 28/30 | Low-proficiency participant; 2 sentences unintelligible |
| 038015 | EIT-1A | 30/30 | |

**117 / 120 sentences transcribed across 4 participants.**

---

## What This Does

Automates transcription of Spanish Elicited Imitation Task (EIT) recordings.

In the EIT, participants listen to 30 sentences of increasing grammatical complexity (7–18 syllables) and repeat back as much as they recall. Their responses are transcribed and scored 0–4 based on meaning preservation. This pipeline automates the transcription step.

---

## Pipeline

```
4 × raw .mp3 recordings
        │
        ▼
speaker_separator.py
   WhisperX large-v3 + pyannote speaker diarization
   Separates the two voices (experimenter + learner)
   → output/<id>/transcript.txt  (timestamped, speaker-labelled)
   → output/<id>/speaker_1.mp3  (isolated audio per speaker)
        │
        ▼
autoeit_transcriber.py
   Finds English/Spanish boundary
   Fuzzy-matches segments to 30 EIT stimuli
   Missing sentences → [no response]
   → AutoEIT_Sample_Audio_for_Transcribing_FILLED.xlsx (column C)
```

---

## Audio Structure

Each recording has this structure — critical for correct processing:

```
[0s – ~74s]     Silence
[~74s – ~121s]  English instructions + practice sentences  ← auto-skipped
[~121s – ~154s] Long silence — English/Spanish boundary
[~154s – end]   30 Spanish EIT trials

  Each trial:
  ┌─────────────────────────────────────────────────┐
  │  Recorded stimulus plays  (~1–4s)               │
  │  Short pause + tone beep                        │
  │  Participant responds     (~0.2–3s)             │
  └─────────────────────────────────────────────────┘
  Inter-trial silence (5–15s)
```

The boundary between English and Spanish is auto-detected by finding the last English-language segment within the first 35% of the recording (via `langdetect`), falling back to the largest silence gap if needed.

---

## How to Run

### Option A — Colab notebook (recommended)

Open `notebooks/AutoEIT.ipynb` in Google Colab, set runtime to **T4 GPU**, and follow sections 1–5.

### Option B — Command line

```bash
# Install (order matters)
pip install torch torchvision torchaudio --upgrade
pip install whisperx
pip install -r requirements.txt

# Stage 1: transcribe + diarise all mp3s in ./input/
python speaker_separator.py

# Stage 2: align to sentences + write Excel
python autoeit_transcriber.py
```

### HuggingFace Token (required for Stage 1)

Speaker diarization uses pyannote, which requires a free HuggingFace token:
1. Create account at huggingface.co
2. Accept the license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Generate a token at huggingface.co/settings/tokens
4. In Colab: add as secret `HF_TOKEN` (🔑 icon in left sidebar)
5. On CLI: `export HF_TOKEN=your_token_here`

---

## Transcription Protocol

Whisper is prompted to follow the official EIT transcription rules:

- Preserve all learner errors — never correct grammar or vocabulary
- `...` for pauses
- `[la-] las` bracket notation for false starts
- English code-switching kept verbatim
- Afterthoughts included as spoken
- Stuttering preserved: `co-co-comerme`
- `[no response]` for silence or unintelligible output
- Prompt-leak guard: segments containing Whisper's own prompt text are automatically discarded

---

## File Structure

```
AutoEIT/
├── README.md
├── speaker_separator.py        ← Stage 1: WhisperX + diarization
├── autoeit_transcriber.py      ← Stage 2: alignment + Excel write
├── requirements.txt
├── notebooks/
│   └── AutoEIT.ipynb           ← self-contained Colab notebook
└── outputs/
    └── AutoEIT_Sample_Audio_for_Transcribing_FILLED.xlsx
```

---
