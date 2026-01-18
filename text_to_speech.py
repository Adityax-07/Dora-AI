import os
import platform
import subprocess
import webbrowser
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs  # pyright: ignore[reportMissingImports]
from gtts import gTTS  # pyright: ignore[reportMissingImports]

load_dotenv()

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY not set")

def play_audio(path):
    os_name = platform.system()
    if os_name == "Windows":
        webbrowser.open(path)
    elif os_name == "Darwin":
        subprocess.run(["afplay", path])
    else:
        subprocess.run(["ffplay", "-autoexit", path])

def text_to_speech_with_elevenlabs(text, output):
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client.text_to_speech.convert(
            text=text,
            voice_id="ZF6FPAbjXT4488VcRRnw",
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32",
        )
        with open(output, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        # Don't play audio here - let Gradio handle it
        return True
    except Exception as e:
        error_str = str(e)
        # Check if it's a quota error
        if "quota_exceeded" in error_str or "401" in error_str or "Unauthorized" in error_str:
            print(f"ElevenLabs quota exceeded, falling back to Google TTS")
            return False
        else:
            # Re-raise other errors
            raise

def text_to_speech_with_gtts(text, output):
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(output)
    # Don't play audio here - let Gradio handle it

def text_to_speech_with_fallback(text, output):
    """
    Try ElevenLabs first, fall back to gTTS if quota exceeded
    """
    success = text_to_speech_with_elevenlabs(text, output)
    if not success:
        # Fallback to Google TTS
        print("Using Google Text-to-Speech as fallback...")
        text_to_speech_with_gtts(text, output)

text = "Hi, I am doing fine, how are you? This is a test for Aditya"

text_to_speech_with_elevenlabs(text, "elevenlabs.mp3")
text_to_speech_with_gtts(text, "gtts.mp3")
