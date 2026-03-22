# AutoEIT — Automated EIT Transcription Pipeline
### GSoC 2026 Evaluation Submission | HumanAI Foundation (NIU + University of Alabama)

**Proposal:** [Audio-to-text transcription for second/additional language learner data](https://humanai.foundation/gsoc/2026/proposal_AutoEIT1.html)  
**Mentors:** Mandy Faretta-Stutenberg (NIU), Xabier Granja (University of Alabama)

---

## Results

| Participant | Sheet | Responses | [no response] | Total filled |
|:---:|---|:---:|:---:|:---:|
| 038010 | 38010-2A | 29 | 1 | 30/30 |
| 038011 | 38011-1A | 30 | 0 | 30/30 |
| 038012 | 38012-2A | 29 | 1 | 30/30 |
| 038015 | 38015-1A | 30 | 0 | 30/30 |

**120 / 120 cells filled. 118 participant responses + 2 `[no response]`.**

Manually reviewed and corrected where noted in cell comments (Excel Review → Comments).

---

## What This Does

Automates transcription of Spanish Elicited Imitation Task (EIT) recordings for Test I.

In the EIT, participants listen to 30 sentences of increasing grammatical complexity (7–18 syllables) and repeat back as much as they recall. Recordings are then transcribed and scored 0–4 based on meaning preservation. This pipeline automates the transcription step.

**The central challenge:** Each recording captures two voices on a single channel — the EIT playback (recorded native-speaker sentences) and the participant's response. Speaker diarization separates them. Fuzzy matching then maps each transcribed segment to the correct sentence number.

---

## Pipeline

```
4 × raw .mp3 recordings
        │
        ▼
speaker_separator.py
   WhisperX large-v3 + pyannote speaker diarization
   → output/<id>/transcript.txt   (timestamped, speaker-labelled)
   → output/<id>/speaker_1.mp3    (isolated audio per speaker)
   → output/<id>/speaker_2.mp3
        │
        ▼
autoeit_transcriber.py
   1. Find English/Spanish boundary (auto-detected via langdetect)
   2. Discard English practice section + instruction segments
   3. Fuzzy-match remaining segments to 30 EIT stimuli
   4. Write responses to column C; missing → [no response]
   → AutoEIT_Sample_Audio_for_Transcribing_FILLED.xlsx
```

---

## Audio Structure

Each recording:

```
[0s – ~74s]     Silence / room noise
[~74s – ~121s]  English instructions + practice  ← auto-skipped
[~121s – ~154s] Long silence (English/Spanish boundary)
[~154s – end]   30 Spanish EIT trials

  Each trial:
  ┌─────────────────────────────────────────────────────┐
  │  Recorded Spanish stimulus plays  (1–4s)            │
  │  Short pause + tone beep                            │
  │  Participant responds             (0.2–3s)          │
  └─────────────────────────────────────────────────────┘
  Inter-trial silence (5–15s)
```

The English/Spanish boundary is auto-detected by finding the last English-language segment within the first 35% of the recording using `langdetect`, with silence-gap fallback.

---

## How to Run

### Option A — Colab Notebook (recommended)

Open `notebooks/AutoEIT.ipynb` in Google Colab → set runtime to **T4 GPU** → run sections 1–5 top to bottom.

### Option B — Command Line

```bash
# Install (order matters)
pip install torch torchvision torchaudio --upgrade
pip install whisperx
pip install -r requirements.txt

# Place mp3 files in ./input/
# Place AutoEIT Sample Audio for Transcribing.xlsx in ./

# Stage 1: transcribe + diarise (~8–12 min per file on GPU)
python speaker_separator.py

# Stage 2: align to sentences + fill Excel (~1 min)
python autoeit_transcriber.py
```

### HuggingFace Token (required for Stage 1)

```bash
# 1. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1
# 2. Generate token: https://huggingface.co/settings/tokens
# 3a. Colab: add as secret named HF_TOKEN (🔑 icon in left sidebar)
# 3b. CLI: export HF_TOKEN=your_token_here
```

---

## Alignment Engine

The sentence alignment in `autoeit_transcriber.py` uses **anchor gravity** — each of the 30 stimuli gets a position bonus (12 / 8 / 4 pts) pulling uncertain matches toward the expected sentence number. This handles:

- `ASSIGN` — solid fuzzy match (score ≥ 40)
- `GUESS` — weak match, position anchor used
- `CONCAT` — continuation of previous response
- `OVERWRITE` — better match than stored
- `REPLACE (RESTART)` — participant restarted the test
- `IGNORE (WEAK)` — discarded (instruction / prompt-leak / too ambiguous)

Split merged segments are recovered by trying `...`-boundary splits first, then brute-forcing word boundaries.

---

## Compatibility Notes

The pipeline runs on current library versions without downgrading anything. Three runtime patches are applied before WhisperX loads (Section 3.1 of the notebook):

- **PyTorch 2.6+** — strict `weights_only` loading disabled for model checkpoints
- **HuggingFace Hub** — deprecated keyword args (`use_auth_token`, `resume_download`) stripped at call time
- **torchaudio** — missing API methods shimmed with `soundfile` equivalents

These run automatically. No manual intervention needed.

---

## Future Work (GSoC Project Plan)

1. **Fine-tune Whisper** on lab-provided corrected EIT transcriptions for lower error on L2 speech
2. **Forced alignment** (Montreal Forced Aligner / WebMAUS) to replace heuristic diarization-based segmentation
3. **Automated scorer** — `scorer.py` assigning 0–4 rubric scores to pre-transcribed responses (Test II)
4. **Confidence-based review queue** — flag low-confidence outputs for human review rather than silent acceptance
5. **Language generalisation** — swap stimulus lists and language models for French, Portuguese, Chinese EIT versions
6. **Researcher UI** — lightweight web interface so non-technical lab members can run the pipeline

---

## File Structure

```
AutoEIT/
├── README.md
├── speaker_separator.py        ← Stage 1: WhisperX + pyannote diarization
├── autoeit_transcriber.py      ← Stage 2: fuzzy alignment + Excel write
├── requirements.txt
├── AutoEIT.ipynb               ← self-contained Colab notebook
└── AutoEIT_Sample_Audio_for_Transcribing_FILLED.xlsx
```

---

## References

- Ortega et al. (2002). Elicited imitation tasks in crosslinguistic SLA research.
- Faretta-Stutenberg et al. (2023). Parallel forms reliability of the Spanish EIT. *RMAL 2*(3). https://doi.org/10.1016/j.rmal.2023.100070
- Bain & Zisserman (2022). WhisperX: Time-Accurate Speech Transcription. https://github.com/m-bain/whisperX
- Plaquet & Bredin (2023). Powerset multi-class cross entropy loss for neural speaker diarization. *Interspeech 2023.*