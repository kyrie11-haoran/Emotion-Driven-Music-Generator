# Emotion-Driven-Music-Generator
an interactive pipeline that turns user text into music. We first classify emotion using a DistilBERT emotion model, map the emotion scores into a structured music prompt, then generate 32kHz audio with MusicGen (Hugging Face), focusing on integration and tuning rather than training from scratch.
