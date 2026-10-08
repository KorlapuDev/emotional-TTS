import os
import argparse
import sys

# Patch to allow PyTorch / torchcodec to find FFmpeg shared DLLs on Windows
ffmpeg_bin = r"C:\Users\Korlapu Abhishek\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build-shared\bin"
if os.path.exists(ffmpeg_bin):
    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ["PATH"]
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(ffmpeg_bin)

from analyzer import analyze_script
from synthesizer import AudioSynthesizer

def main():
    parser = argparse.ArgumentParser(description="Emotion TTS Pipeline")
    parser.add_argument("--script", type=str, default="sample_script.txt", help="Path to input text script")
    parser.add_argument("--reference", type=str, default="reference_voice_clean.wav", help="Path to reference voice WAV")
    parser.add_argument("--output", type=str, default="final_story_narration.wav", help="Path to output audio WAV")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model to use for emotion analysis")
    args = parser.parse_args()

    if not os.path.exists(args.script):
        print(f"Error: Script file '{args.script}' not found.")
        return

    if not os.path.exists(args.reference):
        print(f"Warning: Reference file '{args.reference}' not found.")
        print(f"Please place a 6-10 second clean voice sample at '{args.reference}'.")
        # Proceeding will fail at synthesis, but we can still analyze.

    print("--- Phase 1: Emotion Analysis ---")
    with open(args.script, "r", encoding="utf-8") as f:
        text_content = f.read()

    print("Analyzing script with local LLM...")
    chunks = analyze_script(text_content, model_name=args.model)
    
    if not chunks:
        print("Failed to get analyzed chunks from LLM. Aborting.")
        return

    print(f"Analysis complete. Extracted {len(chunks)} emotional chunks.")
    for c in chunks:
        safe_text = c.get('text', '')[:30].encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(f" - [{c.get('emotion', 'N/A')}] {safe_text}... (Pause: {c.get('pause_after_ms', 0)}ms)")

    if not os.path.exists(args.reference):
        print("Aborting synthesis because reference audio is missing.")
        return

    print("\n--- Phase 2: Speech Synthesis ---")
    synth = AudioSynthesizer()
    synth.generate_narration(chunks, args.reference, args.output)

if __name__ == "__main__":
    main()
