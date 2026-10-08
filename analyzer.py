import json
import ollama

def analyze_script(text_content: str, model_name: str = "llama3.2:3b") -> list[dict]:
    """
    Analyzes the script and breaks it down into sentences with emotion, speed, and pacing tags.
    """
    system_prompt = """You are a script analyzer for an anime recap narration voiceover.
Analyze the input text sentence by sentence.
Tasks:
1. Detect emotional tone: (e.g., nostalgic, tense, calm, dramatic).
2. Assign pacing parameters: speed multiplier (0.85 to 1.10).
3. CRITICAL: You MUST transliterate the Romanized Hinglish text into pure Devanagari (Hindi) script. The "text" field in the output MUST be written in Devanagari. Example: "Hero" -> "हीरो", "Demon King" -> "डीमन किंग", "kaam" -> "काम".
4. CRITICAL: Keep each chunk SHORT (maximum 8-12 words). If a sentence is long, split it into multiple consecutive chunks. Long chunks cause severe audio distortion!

Output valid JSON only containing an object with a single key "chunks", which is a list of objects with the following keys:
- sentence_id (integer)
- text (string - MUST BE IN DEVANAGARI SCRIPT)
- emotion (string)
- speed (float)
- pause_after_ms (integer)

Example Output:
{
  "chunks": [
    {
      "sentence_id": 1,
      "text": "क्या होता है जब हीरो डीमन किंग को हरा देता है? कहानियाँ वहीं खत्म हो जाती हैं...",
      "emotion": "reflective",
      "speed": 0.92,
      "pause_after_ms": 450
    }
  ]
}
"""
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_content}
        ],
        format="json",
        options={"temperature": 0.3}
    )
    
    try:
        content = response['message']['content']
        parsed = json.loads(content)
        
        # We expect {"chunks": [...]}
        if isinstance(parsed, dict) and "chunks" in parsed:
            return parsed["chunks"]
            
        # Fallbacks:
        # Sometimes models might return a dict like {"sentences": [...]}
        if isinstance(parsed, dict) and len(parsed.keys()) == 1:
            key = list(parsed.keys())[0]
            if isinstance(parsed[key], list):
                return parsed[key]
                
        # If the model returned a single dictionary (one sentence), wrap it in a list
        if isinstance(parsed, dict):
            return [parsed]
            
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON list from the model.")
        return parsed
    except Exception as e:
        print(f"Failed to parse LLM response: {e}")
        print("Raw response:", response['message']['content'])
        return []
