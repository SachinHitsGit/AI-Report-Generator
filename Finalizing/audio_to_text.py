import whisper

# Load Whisper model
print("📦 Loading model...")
model = whisper.load_model("base")  # use "small" or "base" for CPU-only

# Transcribe audio
print("🚀 Starting transcription...")
result = model.transcribe("meeting.wav")

# Save plain text transcript
with open("meeting.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("✅ Done! Transcript saved to 'meeting.txt'")
