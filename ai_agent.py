import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from tool import analyze_image_with_query

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable must be set")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=groq_api_key,
)

def ask_agent(user_query: str) -> str:
    try:
        # Check if query requires vision
        vision_keywords = ["see", "look", "camera", "what", "show", "image", "photo", "picture", "visual", "how many", "describe"]
        if any(keyword in user_query.lower() for keyword in vision_keywords):
            # Call vision tool
            response = analyze_image_with_query(user_query)
            return response
        else:
            # Direct LLM response
            response = llm.invoke(user_query)
            return response.content
    except Exception as e:
            error_msg = str(e)
            if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                return f"Error: API quota exceeded. Please wait a few minutes and try again."
            raise
    
    
    if __name__ == "__main__":
        text = input("Enter text to convert to speech: ").strip()
        if not text:
            raise ValueError("Text cannot be empty")
    
        text_to_speech_with_elevenlabs(text, "elevenlabs.mp3")
        text_to_speech_with_gtts(text, "gtts.mp3")



