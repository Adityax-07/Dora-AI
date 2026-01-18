import cv2
import base64
from dotenv import load_dotenv

load_dotenv()

def capture_image() -> str:
    """
    Captures one frame from the default webcam, resizes it,
    encodes it as Base64 JPEG (raw string) and returns it.
    """
    for idx in range(4):
        # Try different backends for Windows compatibility
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # DirectShow backend for Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)  # Fallback to default backend
        if cap.isOpened():
            for _ in range(10):  # Warm up
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            cv2.imwrite("sample.jpg", frame)  # Optional
            ret, buf = cv2.imencode('.jpg', frame)
            if ret:
                return base64.b64encode(buf).decode('utf-8')
    raise RuntimeError(
        "Could not open any webcam (tried indices 0-3). "
        "Please ensure:\n"
        "1. A webcam is connected to your computer\n"
        "2. No other application is using the webcam\n"
        "3. Webcam drivers are properly installed"
    )


import os
from groq import Groq

def analyze_image_with_query(query: str) -> str:
    """
    Expects a string with 'query'.
    Captures the image and sends the query and the image to
    to Groq's vision chat API and returns the analysis.
    """
    img_b64 = capture_image()
    model="meta-llama/llama-4-maverick-17b-128e-instruct"
    
    if not query or not img_b64:
        return "Error: both 'query' and 'image' fields required."

    # Load Groq API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_api_key_here":
        return "Error: GROQ_API_KEY not set in .env file. Please add your Groq API key."
    
    client = Groq(api_key=groq_api_key)  
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                    },
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content

#query = "How many people do you see?"
#print(analyze_image_with_query(query))