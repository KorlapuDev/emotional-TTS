import os
import torch
from TTS.api import TTS
from pydub import AudioSegment

class AudioSynthesizer:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing TTS on {self.device}...")
        # Using fp16 to save VRAM (requires GPU)
        gpu_flag = True if self.device == "cuda" else False
        self.tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu_flag)
        
    def generate_narration(self, processed_chunks: list[dict], speaker_wav: str, output_path: str = "final_story_narration.wav"):
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Reference voice file not found: {speaker_wav}")

        audio_segments = []
        temp_files = []

        print("Starting voice generation loop...")
        for chunk in processed_chunks:
            # Skip invalid chunks
            if "text" not in chunk or not chunk["text"].strip():
                continue

            sentence_id = chunk.get("sentence_id", "unk")
            text = chunk["text"]
            emotion = chunk.get("emotion", "neutral")
            speed = chunk.get("speed", 1.0)
            pause_after_ms = chunk.get("pause_after_ms", 500)
            
            import sys
            safe_text = text[:50].encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
            print(f"[{emotion.upper()}] Synthesizing: {safe_text}... (Speed: {speed})")

            out_path = f"temp_segment_{sentence_id}.wav"
            temp_files.append(out_path)
            
            # The speed param might need adjustment depending on the TTS version. 
            # XTTSv2 supports speed parameter in some implementations but if it throws an error we can handle it.
            try:
                self.tts.tts_to_file(
                    text=text,
                    file_path=out_path,
                    speaker_wav=speaker_wav,
                    language="hi",
                    speed=speed,
                    split_sentences=False
                )
            except TypeError:
                print("Warning: Speed parameter not supported natively by this TTS version. Falling back to default speed.")
                self.tts.tts_to_file(
                    text=text,
                    file_path=out_path,
                    speaker_wav=speaker_wav,
                    language="hi",
                    split_sentences=False
                )
            
            segment = AudioSegment.from_wav(out_path)
            audio_segments.append(segment)
            
            if pause_after_ms > 0:
                audio_segments.append(AudioSegment.silent(duration=pause_after_ms))

        # Combine all segments with crossfades to prevent clicks
        print("Stitching segments together...")
        if not audio_segments:
            final_audio = AudioSegment.silent(duration=0)
        else:
            final_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                # Use a small 30ms crossfade to smooth out hard cuts/clicks between generations
                final_audio = final_audio.append(segment, crossfade=30)
        
        # We skip aggressive dB normalization to prevent amplifying the static/noise floor
        
        final_audio.export(output_path, format="wav")
        print(f"Final audio saved to {output_path}")
        
        # Clean up temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _normalize(self, audio_segment, target_dBFS):
        """Normalize audio to a target dBFS."""
        change_in_dBFS = target_dBFS - audio_segment.dBFS
        return audio_segment.apply_gain(change_in_dBFS)
