# Emotion-Driven Music Generator 🎵🎭

An interactive pipeline that transforms natural language input into original music. We classify emotion from text using state-of-the-art NLP models, map the emotion scores into a structured music prompt, and generate 32 kHz audio with MusicGen (Hugging Face) — focusing on integration and fine-tuning rather than training from scratch.

---

## 📁 Repository Structure

```
Emotion-Driven-Music-Generator/
│
├── Demo_emotion_music.ipynb                                        # Simple Gradio demo: text → emotion → music
├── Emotion_conditioned_music_generation_with_semantic_alignment.ipynb  # Advanced 4-model pipeline with semantic alignment
├── Cinematic_Scene_to_Score_Generator.ipynb                       # Scene description → cinematic score + MIDI/MusicXML export
├── multilingual_audio anaylsis_richer emotion-to-music project.ipynb  # Multilingual speech/text → emotion → music
├── ABC_notes_gen_LLM.ipynb                                        # GPT-style transformer trained from scratch on ABC music notation
│
├── FinalPresentation.pdf
├── Technical Topic Presentation.pdf
├── Project Preview Presentation.pdf
└── Project plan presentation.pdf
```

---

## 🗂️ Notebook Descriptions

### 1. `Demo_emotion_music.ipynb` — Simple Emotion-to-Music Demo
A clean, minimal demo with a Gradio web interface. Enter any text, and the pipeline:
1. Classifies 28 emotions using **RoBERTa** (`SamLowe/roberta-base-go_emotions`)
2. Maps top emotions to a descriptive music prompt
3. Generates audio using **MusicGen** (`facebook/musicgen-small`)

**Best for:** Quick exploration and demonstrations.

---

### 2. `Emotion_conditioned_music_generation_with_semantic_alignment.ipynb` — Advanced Pipeline ⭐
*"Catharsis: Your Feelings Deserve a Soundtrack"*

A 4-model pipeline featuring a semantic alignment feedback loop and explainability tools:
1. **Emotion extraction** with uncertainty quantification (entropy) — RoBERTa
2. **Music prompt generation** via LLM with self-evaluation — Flan-T5
3. **Semantic alignment check** between prompt and emotion — Sentence-Transformers (`all-MiniLM-L6-v2`)
4. **Audio generation** — MusicGen
5. Visualizations: emotion radar charts, token attribution
6. **Fréchet Audio Distance (FAD)** metric for audio quality evaluation

**Best for:** Research, experimentation, and explainability.

---

### 3. `Cinematic_Scene_to_Score_Generator.ipynb` — Scene-to-Score Generator
Generate cinematic film scores from text descriptions of movie scenes. The pipeline:
1. Parses scene text for emotions and cinematic context (scene type, intensity)
2. Builds a tailored music prompt
3. Generates audio using **MusicGen**
4. Composes a symbolic score (notes, tempo, key, instruments) using **music21**
5. Exports **MIDI** and **MusicXML** files for use in a DAW or notation software

**Best for:** Film scoring, game audio, creative projects.

---

### 4. `multilingual_audio anaylsis_richer emotion-to-music project.ipynb` — Multilingual Audio Analysis
Supports text or speech input in 200+ languages:
1. Transcribes audio using **Whisper** (`openai-whisper`)
2. Detects language with **XLM-RoBERTa**
3. Translates to English using **NLLB-200** (`facebook/nllb-200-distilled-600M`)
4. Classifies emotions and generates music via MusicGen

**Best for:** Multilingual content, dialogue-based applications (movies, podcasts, games).

---

### 5. `ABC_notes_gen_LLM.ipynb` — GPT-Style Music Notation Model
An educational implementation of a character-level GPT transformer trained from scratch (following Andrej Karpathy's "Zero To Hero" series) on Irish folk music in **ABC notation**:
1. Loads an ABC notation dataset from Hugging Face
2. Builds and trains a custom transformer (multi-head attention, feed-forward blocks)
3. Generates new ABC notation sequences

**Best for:** Learning transformer fundamentals applied to music generation.

---

## ⚙️ Requirements

All notebooks are designed to run in **Google Colab** (recommended) or a local Python environment with GPU support.

### Common Dependencies

| Package | Purpose |
|---|---|
| `torch` / `torchaudio` | Deep learning framework |
| `transformers` | Pre-trained models (RoBERTa, MusicGen, Flan-T5, etc.) |
| `accelerate` | Efficient model loading |
| `gradio` | Interactive web UI |
| `scipy` | Audio file I/O |
| `datasets` | Hugging Face datasets |

### Notebook-Specific Dependencies

| Notebook | Extra Packages |
|---|---|
| Semantic Alignment | `sentence-transformers`, `matplotlib` |
| Cinematic Generator | `music21` |
| Multilingual | `openai-whisper`, `sentencepiece`, `ffmpeg` (system) |

---

## 🚀 How to Run

### Option A: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook** and select the `.ipynb` file you want to run
3. Set the runtime to **GPU**: `Runtime → Change runtime type → T4 GPU`
4. Run the first cell (installs dependencies automatically)
5. Run all remaining cells in order (`Runtime → Run all`)

> **Note:** The first run will download model weights (~1–3 GB). Subsequent runs in the same session are faster.

---

### Option B: Local Setup

#### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (strongly recommended; CPU is very slow for audio generation)
- `git`

#### Steps

```bash
# 1. Clone the repository
git clone https://github.com/kyrie11-haoran/Emotion-Driven-Music-Generator.git
cd Emotion-Driven-Music-Generator

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install core dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate gradio scipy datasets

# 4. Install notebook-specific extras as needed:

# For Demo_emotion_music.ipynb (no extras needed beyond core)

# For Emotion_conditioned_music_generation_with_semantic_alignment.ipynb
pip install sentence-transformers matplotlib

# For Cinematic_Scene_to_Score_Generator.ipynb
pip install music21

# For multilingual_audio anaylsis_richer emotion-to-music project.ipynb
pip install openai-whisper sentencepiece
sudo apt-get install -y ffmpeg   # Linux; use Homebrew on macOS

# For ABC_notes_gen_LLM.ipynb
pip install datasets

# 5. Launch Jupyter
jupyter notebook
```

Then open the desired `.ipynb` file from the Jupyter file browser.

---

### Per-Notebook Quick Start

| Notebook | Minimum Install Command |
|---|---|
| `Demo_emotion_music.ipynb` | `pip install transformers accelerate gradio scipy torch torchaudio` |
| `Emotion_conditioned_...ipynb` | `pip install transformers accelerate gradio scipy sentence-transformers matplotlib torch torchaudio` |
| `Cinematic_Scene_to_Score_Generator.ipynb` | `pip install transformers accelerate gradio scipy music21 torch torchaudio` |
| `multilingual_audio...ipynb` | `pip install transformers accelerate gradio scipy openai-whisper sentencepiece torch torchaudio` + `ffmpeg` |
| `ABC_notes_gen_LLM.ipynb` | `pip install torch datasets` |

---

## 🧠 Models Used

| Model | Source | Used In |
|---|---|---|
| `facebook/musicgen-small` | Hugging Face | All audio-generation notebooks |
| `SamLowe/roberta-base-go_emotions` | Hugging Face | Demo, Semantic, Multilingual |
| `google/flan-t5-base` | Hugging Face | Semantic Alignment notebook |
| `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face | Semantic Alignment notebook |
| `papluca/xlm-roberta-base-language-detection` | Hugging Face | Multilingual notebook |
| `facebook/nllb-200-distilled-600M` | Hugging Face | Multilingual notebook |
| `openai/whisper-base` | OpenAI via pip | Multilingual notebook |
| Custom GPT Transformer | Built from scratch | ABC Notes notebook |

---

## 📌 Tips

- **GPU memory:** `musicgen-small` requires ~2 GB VRAM. If you hit OOM errors, reduce `max_new_tokens` or switch to CPU.
- **First run:** Model downloads may take several minutes depending on your connection.
- **Gradio UI:** After running the final cell in any notebook with a Gradio demo, a local URL (and optionally a public share link) will appear in the output.
- **Audio output:** Generated audio is saved as `.wav` files and also streamed through the Gradio player.

---

## 📄 License

This project was developed for academic purposes. See individual model licenses on Hugging Face for usage restrictions.

