import logging
import speech_recognition as sr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Function to record audio from the microphone and save it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            # Record the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Save as WAV (no ffmpeg required) - Groq Whisper accepts WAV format
            wav_data = audio_data.get_wav_data()
            # If file_path ends with .mp3, change to .wav
            if file_path.endswith('.mp3'):
                file_path = file_path[:-4] + '.wav'
            
            with open(file_path, "wb") as f:
                f.write(wav_data)
            
            logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")



import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def transcribe_with_groq(audio_filepath):
    """
    Transcribe audio file using Groq's Whisper model.
    
    Args:
        audio_filepath (str): Path to the audio file to transcribe.
    
    Returns:
        str: The transcribed text.
    
    Raises:
        ValueError: If GROQ_API_KEY is not set.
        FileNotFoundError: If the audio file doesn't exist.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment. Please check your .env file.")
    
    if not os.path.exists(audio_filepath):
        raise FileNotFoundError(f"Audio file not found: {audio_filepath}")
    
    client = Groq(api_key=GROQ_API_KEY)
    stt_model = "whisper-large-v3"
    
    try:
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        return transcription.text
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        raise

# Example usage (only run when script is executed directly, not imported)
if __name__ == "__main__":
    audio_filepath = "test_speech_to_text.wav"
    record_audio(audio_filepath)
    print(transcribe_with_groq(audio_filepath))