"""
Speaker Separator using WhisperX  — Batch Folder Mode
───────────────────────────────────────────────────────
Usage:
    python speaker_separator.py --folder ./my_audio_files/
    python speaker_separator.py --folder ./my_audio_files/ --output ./results/

Requirements:
    pip install whisperx pydub
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# ── MASTER PATCH BLOCK (MUST RUN BEFORE WHISPERX) ────────────────────────────
import torch
import torchaudio
import huggingface_hub
import soundfile as sf

# 0. PyTorch 2.6+ Strict Loading Bypass (Double-Run Proof!)
if not getattr(torch, '_load_is_patched', False):
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    torch._load_is_patched = True

_orig_hf_hub_download = huggingface_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs: kwargs['token'] = kwargs.pop('use_auth_token')
    kwargs.pop('resume_download', None)
    kwargs.pop('force_filename', None)
    kwargs.pop('local_dir_use_symlinks', None)
    try: return _orig_hf_hub_download(*args, **kwargs)
    except Exception as e:
        if "custom.py" in str(kwargs.get('filename', '')) and ("404" in str(e) or "EntryNotFound" in type(e).__name__):
            raise ValueError("File not found")
        raise e
huggingface_hub.hf_hub_download = _patched_hf_hub_download

_orig_model_info = huggingface_hub.model_info
def _patched_model_info(*args, **kwargs):
    if 'use_auth_token' in kwargs: kwargs['token'] = kwargs.pop('use_auth_token')
    return _orig_model_info(*args, **kwargs)
huggingface_hub.model_info = _patched_model_info

class DummyAudioMetaData:
    def __init__(self, sr, frames, ch):
        self.sample_rate = sr; self.num_frames = frames; self.num_channels = ch
def _patched_info(uri, **kwargs):
    info = sf.info(uri)
    return DummyAudioMetaData(info.samplerate, info.frames, info.channels)
def _patched_load(uri, **kwargs):
    data, sr = sf.read(uri, always_2d=True, dtype='float32')
    return torch.from_numpy(data).t(), sr
def _patched_save(filepath, src, sample_rate, **kwargs):
    sf.write(filepath, src.t().numpy(), sample_rate)

if not hasattr(torchaudio, 'AudioMetaData'): torchaudio.AudioMetaData = DummyAudioMetaData
if not hasattr(torchaudio, 'list_audio_backends'): torchaudio.list_audio_backends = lambda: ['soundfile', 'sox_io']
if not hasattr(torchaudio, 'info'): torchaudio.info = _patched_info
if not hasattr(torchaudio, 'load'): torchaudio.load = _patched_load
if not hasattr(torchaudio, 'save'): torchaudio.save = _patched_save
# ═════════════════════════════════════════════════════════════════════════════from whisperx.diarize import DiarizationPipeline
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)# ─── HF TOKEN — tries Colab secrets first, then env var, then hardcoded ───────
def load_hf_token():
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            print(f"  ✅ HF token loaded from Colab secrets: {token[:8]}...")
            return token
    except Exception:
        pass
    token = os.environ.get('HF_TOKEN', '')
    if token:
        print(f"  ✅ HF token loaded from environment: {token[:8]}...")
        return token
    # Fallback — hardcode here if needed
    hardcoded = ""   # ← paste token here as last resort
    if hardcoded:
        print(f"  ✅ HF token loaded from hardcoded value: {hardcoded[:8]}...")
        return hardcoded
    print("  ❌ HF_TOKEN not found in Colab secrets, environment, or hardcoded value.")
    return ""# ─── CONFIG ───────────────────────────────────────────────────────────────────

INPUT_FOLDER  = "./audio_files"
OUTPUT_DIR    = "speaker_output"
# WHISPER_MODEL = "large"
WHISPER_MODEL = "large-v3"
TARGET_DBFS   = -20.0
LANGUAGE      = "es"

INITIAL_PROMPT = (
"""
    Un estudiante de español L2 repite una frase.
Transcribe exactamente lo que dice.
Incluye errores gramaticales, disfluencias (eh, mm, um),
falsos inicios con guión (ej: la- las), y pausas con '...'.
Si hay una pausa demasiado larga, probablemente indica el inicio de la siguiente oración.
No corrijas ni normalices el texto.
No traduzcas ninguna frase en español al inglés."""
)

# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Speaker 1": "\033[94m",
    "Speaker 2": "\033[92m",
    "Speaker 3": "\033[93m",
    "Speaker 4": "\033[95m",
    "reset":     "\033[0m",
    "bold":      "\033[1m",
    "dim":       "\033[2m",
    "header":    "\033[96m",
    "green":     "\033[92m",
    "red":       "\033[91m",
    "yellow":    "\033[93m",
}

def color(text, *keys):
    codes = "".join(COLORS.get(k, "") for k in keys)
    return f"{codes}{text}{COLORS['reset']}"

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def print_segment(speaker, start, end, text, index):
    spk_color = COLORS.get(speaker, "\033[97m")
    ts  = color(f"[{format_time(start)} → {format_time(end)}]", "dim")
    spk = f"{spk_color}{COLORS['bold']}{speaker:<10}{COLORS['reset']}"
    idx = color(f"#{index:03d}", "dim")
    print(f"  {idx}  {ts}  {spk}  {text}")

def print_header(title):
    width = 70
    print("\n" + color("─" * width, "header"))
    print(color(f"  {title}", "header", "bold"))
    print(color("─" * width, "header"))

def print_banner(title):
    width = 70
    print("\n" + color("═" * width, "bold"))
    print(color(f"  {title}", "bold"))
    print(color("═" * width, "bold"))

def check_requirements():
    missing = []
    for pkg, imp in [("whisperx", "whisperx"), ("pydub", "pydub")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(color(f"\n❌ Missing packages. Run:\n\n   pip install {' '.join(missing)}\n", "bold"))
        sys.exit(1)def process_file(mp3_path, file_output_dir, whisper_model, align_model,
                 metadata, diarize_model, device, token):
    from pydub import AudioSegment
    from pydub.effects import normalize
    from collections import defaultdict
    import whisperx
    import subprocess # <-- NEW IMPORT

    os.makedirs(file_output_dir, exist_ok=True)
    wav_path = os.path.join(file_output_dir, "_working.wav")

    print(color("  ⏳ Leveling audio dynamically (FFmpeg dynaudnorm)…", "dim"))

    # ── THE FIX: Dynamic Audio Normalizer ──
    # f=150: Uses a fast 150ms sliding window to catch quick pauses and drop-offs.
    # m=15: Allows it to boost quiet sections (like trail-offs) by up to 15x.
    # This is magic for speech: it keeps sharp words intact while bringing whispers to the front.
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", mp3_path,
        "-af", "dynaudnorm=f=150:m=15",
        wav_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg dynamic normalization failed: {e}")

    # Load the dynamically leveled wav file back into PyDub
    audio_pydub = AudioSegment.from_wav(wav_path)
    total_duration = len(audio_pydub) / 1000

    print(f"     ✅ Audio dynamically leveled ({total_duration:.1f}s)")
    # ────────────────────────────────────────

    audio_array = whisperx.load_audio(wav_path)

    print(color("  ⏳ Transcribing…", "dim"))
    transcribe_kwargs = {"batch_size": 16}
    if LANGUAGE:
        transcribe_kwargs["language"] = LANGUAGE
    result   = whisper_model.transcribe(audio_array, **transcribe_kwargs)
    language = result.get("language", LANGUAGE or "en")
    print(f"     ✅ Transcription done  (language: {language})")

    print(color("  ⏳ Aligning word timestamps…", "dim"))
    if metadata.get("language") != language:
        print(f"     ⚠️  Language mismatch ({metadata.get('language')} → {language}), reloading align model…")
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        metadata["language"] = language
    result = whisperx.align(result["segments"], align_model, metadata, audio_array, device)
    print("     ✅ Alignment done")

    print(color("  ⏳ Diarizing speakers…", "dim"))
    diarize_segs = diarize_model(audio_array, min_speakers=2, max_speakers=2)
    result       = whisperx.assign_word_speakers(diarize_segs, result)
    print("     ✅ Diarization done")

    raw_speakers = sorted({
        seg.get("speaker", "UNKNOWN")
        for seg in result["segments"] if seg.get("speaker")
    })
    speaker_map = {raw: f"Speaker {i+1}" for i, raw in enumerate(raw_speakers)}

    segments = []
    for seg in result["segments"]:
        raw = seg.get("speaker", "UNKNOWN")
        segments.append({
            "speaker":  speaker_map.get(raw, "Unknown"),
            "start":    seg["start"],
            "end":      seg["end"],
            "start_ms": int(seg["start"] * 1000),
            "end_ms":   int(seg["end"]   * 1000),
            "text":     seg["text"].strip(),
        })

    print_header(f"  SEGMENTS  ({len(segments)} total)")
    for i, seg in enumerate(segments, 1):
        print_segment(seg["speaker"], seg["start"], seg["end"], seg["text"], i)

    print_header("  SPEAKER SUMMARY")
    spk_stats = defaultdict(lambda: {"count": 0, "duration": 0.0})
    for seg in segments:
        spk_stats[seg["speaker"]]["count"]    += 1
        spk_stats[seg["speaker"]]["duration"] += seg["end"] - seg["start"]

    for spk, stats in sorted(spk_stats.items()):
        spk_color = COLORS.get(spk, "\033[97m")
        bar_len   = int(stats["duration"] / total_duration * 40)
        bar       = "█" * bar_len + "░" * (40 - bar_len)
        print(f"\n  {spk_color}{COLORS['bold']}{spk}{COLORS['reset']}")
        print(f"    Segments : {stats['count']}")
        print(f"    Duration : {stats['duration']:.1f}s  "
              f"({stats['duration'] / total_duration * 100:.1f}%)")
        print(f"    {spk_color}{bar}{COLORS['reset']}")

    transcript_path = os.path.join(file_output_dir, "transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("        SPEAKER-SEPARATED TRANSCRIPT (WhisperX)\n")
        f.write(f"        Source: {os.path.basename(mp3_path)}\n")
        f.write("=" * 70 + "\n")

        current_speaker = None
        for seg in segments:
            if seg["speaker"] != current_speaker:
                current_speaker = seg["speaker"]
                f.write(f"\n{'─'*70}\n{current_speaker}:\n")
            f.write(f"  [{format_time(seg['start'])} → {format_time(seg['end'])}]  {seg['text']}\n")

        f.write(f"\n{'='*70}\nSPEAKER STATS\n{'='*70}\n")
        for spk, stats in sorted(spk_stats.items()):
            f.write(f"  {spk}: {stats['count']} segments, {stats['duration']:.1f}s "
                    f"({stats['duration']/total_duration*100:.1f}%)\n")

    print_header("  SAVING FILES")
    print(f"\n  ✅ Transcript  →  {transcript_path}")

    speaker_audio = {}
    for seg in segments:
        spk   = seg["speaker"]
        chunk = audio_pydub[seg["start_ms"]:seg["end_ms"]]
        if spk not in speaker_audio:
            speaker_audio[spk] = AudioSegment.silent(duration=0)
        speaker_audio[spk] += chunk

    for spk, clip in sorted(speaker_audio.items()):
        fname     = f"{spk.replace(' ', '_').lower()}.mp3"
        clip_path = os.path.join(file_output_dir, fname)

        # Normalize the isolated output file to max volume without clipping
        if len(clip) > 0:
            clip = normalize(clip)

        clip.export(clip_path, format="mp3")
        print(f"  ✅ {spk} audio  →  {clip_path}  ({len(clip)/1000:.1f}s)")

    os.remove(wav_path)
    return len(segments)def main():
    check_requirements()
    input_folder = "./input"
    output_root  = "./output"

    import torch
    import whisperx

    # ── Token — Colab → env var → hardcoded ───────────────────────────────────
    token = load_hf_token()
    if not token:
        sys.exit(1)

    if not os.path.isdir(input_folder):
        print(color(f"\n❌ Folder not found: {input_folder}\n", "bold"))
        sys.exit(1)

    mp3_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(".mp3")
    ])

    if not mp3_files:
        print(color(f"\n⚠️  No MP3 files found in: {input_folder}\n", "yellow"))
        sys.exit(0)

    os.makedirs(output_root, exist_ok=True)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"

    print_banner("SPEAKER SEPARATOR — WhisperX  (Batch Mode)")
    print(f"\n  📂 Input folder : {input_folder}")
    print(f"  📁 Output root  : {output_root}/")
    print(f"  🎵 Files found  : {len(mp3_files)}")
    print(f"  🖥  Device       : {device.upper()}  |  Model : {WHISPER_MODEL}")
    print(f"  🌐 Language     : {LANGUAGE or 'auto-detect'}")
    print(f"\n  Files to process:")
    for i, f in enumerate(mp3_files, 1):
        print(f"    {i}. {os.path.basename(f)}")

    print_header("LOADING MODELS  (once for all files)")

    print(color("\n  ⏳ Loading Whisper model…", "dim"))
    asr_options   = {"initial_prompt": INITIAL_PROMPT} if INITIAL_PROMPT else {}
    whisper_model = whisperx.load_model(WHISPER_MODEL, device, compute_type=compute,
                                        asr_options=asr_options)
    if INITIAL_PROMPT:
        print(f"     📝 Initial prompt active ({len(INITIAL_PROMPT)} chars)")
    print("     ✅ Whisper ready")

    print(color("  ⏳ Loading alignment model…", "dim"))
    align_lang            = LANGUAGE if LANGUAGE else "en"
    align_model, metadata = whisperx.load_align_model(language_code=align_lang, device=device)
    metadata["language"]  = align_lang
    print(f"     ✅ Alignment model ready  (language: {align_lang})")

    print(color("  ⏳ Loading diarization pipeline…", "dim"))
    diarize_model = DiarizationPipeline(token=token, device=device)
    print("     ✅ Diarization pipeline ready")

    results_summary = []

    for i, mp3_path in enumerate(mp3_files, 1):
        filename    = os.path.splitext(os.path.basename(mp3_path))[0]
        file_output = os.path.join(output_root, filename)

        print_banner(f"FILE {i}/{len(mp3_files)}  —  {os.path.basename(mp3_path)}")

        try:
            n_segments = process_file(
                mp3_path, file_output, whisper_model,
                align_model, metadata, diarize_model, device, token
            )
            results_summary.append((os.path.basename(mp3_path), "✅", n_segments, file_output))
            print(color(f"\n  ✅ Done  →  {file_output}/", "green", "bold"))

        except Exception as e:
            results_summary.append((os.path.basename(mp3_path), "❌", 0, str(e)))
            print(color(f"\n  ❌ Failed: {e}", "red", "bold"))

    print_banner("BATCH COMPLETE 🎉")
    print(f"\n  {'FILE':<35}  {'STATUS':<6}  SEGMENTS / OUTPUT")
    print(f"  {'─'*35}  {'─'*6}  {'─'*30}")
    for fname, status, n, out in results_summary:
        info = f"{n} segments  →  {out}" if status == "✅" else out
        print(f"  {fname:<35}  {status:<6}  {info}")

    passed = sum(1 for _, s, _, _ in results_summary if s == "✅")
    print(f"\n  {passed}/{len(mp3_files)} files processed successfully.\n")


if __name__ == "__main__":
    main()