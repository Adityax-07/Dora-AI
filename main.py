import os
import time
import gradio as gr
from speech_to_text import record_audio, transcribe_with_groq
from ai_agent import ask_agent
from text_to_speech import text_to_speech_with_elevenlabs, text_to_speech_with_gtts, text_to_speech_with_fallback, play_audio

# Set custom CA bundle for HTTPS requests
os.environ['REQUESTS_CA_BUNDLE'] = 'root_ca.pem'

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
audio_filepath = "audio_question.wav"

def start_conversation():
    global conversation_active
    conversation_active = True
    return "🎤 Conversation started - Listening..."

def pause_conversation():
    global conversation_active
    conversation_active = False
    return "⏸️ Conversation paused"

def end_conversation():
    global conversation_active
    conversation_active = False
    return "⏹️ Conversation ended"

def process_audio_and_chat(chat_history):
    global conversation_active
    
    if not conversation_active:
        time.sleep(0.5)
        return chat_history, None, "⏸️ Paused - Press '▶️ Start' to continue"
    
    try:
        record_audio(file_path=audio_filepath)
        user_input = transcribe_with_groq(audio_filepath)
        
        # Skip if transcription is empty or too short (likely noise/silence)
        if not user_input or len(user_input.strip()) < 3:
            print("No speech detected, waiting for next input...")
            return chat_history, None, "🎤 Listening... (no speech detected)"

        response = ask_agent(user_query=user_input)

        output_filepath = f"final_{int(time.time())}.mp3"
        text_to_speech_with_fallback(text=response, output=output_filepath)

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})

        return chat_history, output_filepath, "🎤 Listening..."

    except Exception as e:
        print(f"Error in recording: {e}")
        time.sleep(2)
        return chat_history, None, f"⚠️ Error: {str(e)}"

# Code for frontend
import cv2
# Global variables
camera = None
is_running = False
last_frame = None
conversation_active = False

def initialize_camera():
    """Initialize the camera with optimized settings"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            # Optimize camera settings for better performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
    return camera is not None and camera.isOpened()

def start_webcam():
    """Start the webcam feed"""
    global is_running, last_frame
    is_running = True
    if not initialize_camera():
        return None
    
    ret, frame = camera.read()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        return frame
    return last_frame

def stop_webcam():
    """Stop the webcam feed"""
    global is_running, camera
    is_running = False
    if camera is not None:
        camera.release()
        camera = None
    return None

def get_webcam_frame():
    """Get current webcam frame with optimized performance"""
    global camera, is_running, last_frame
    
    if not is_running or camera is None:
        return last_frame
    
    # Skip frames if buffer is full to avoid lag
    if camera.get(cv2.CAP_PROP_BUFFERSIZE) > 1:
        for _ in range(int(camera.get(cv2.CAP_PROP_BUFFERSIZE)) - 1):
            camera.read()
    
    ret, frame = camera.read()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        return frame
    return last_frame

# Setup UI

with gr.Blocks() as demo:
    # Header with avatar
    with gr.Row():
        gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: center; padding: 20px;">
                <img src="file/dora_avatar.jpg" style="width: 80px; height: 80px; border-radius: 50%; margin-right: 20px; object-fit: cover;" />
                <h1 style="color: orange; margin: 0; font-size: 3em;">Dora – Your Personal AI Assistant</h1>
            </div>
        """)

    with gr.Row():
        # Left column - Webcam
        with gr.Column(scale=1):
            gr.Markdown("## Webcam Feed")
            
            with gr.Row():
                start_btn = gr.Button("Start Camera", variant="primary")
                stop_btn = gr.Button("Stop Camera", variant="secondary")
            
            webcam_output = gr.Image(
                label="Live Feed",
                streaming=True,
                show_label=False,
                width=640,
                height=480
            )
            
            # Faster refresh rate for smoother video
            webcam_timer = gr.Timer(0.033)  # ~30 FPS (1/30 ≈ 0.033 seconds)
        
        # Right column - Chat
        with gr.Column(scale=1):
            gr.Markdown("## Chat Interface")
            
            # Conversation control buttons
            with gr.Row():
                start_conv_btn = gr.Button("▶️ Start", variant="primary", scale=1)
                pause_conv_btn = gr.Button("⏸️ Pause", variant="secondary", scale=1)
                end_conv_btn = gr.Button("⏹️ End", variant="stop", scale=1)
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)
            
            # Status indicator
            status_text = gr.Textbox(
                value="⏸️ Press '▶️ Start' to begin conversation",
                label="Status",
                interactive=False,
                show_label=False
            )
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=350,
                show_label=False
            )
            
            audio_output = gr.Audio(
                label="AI Response Audio",
                autoplay=True
            )
    
    # Conversation timer (runs when conversation is active)
    conversation_timer = gr.Timer(2)  # Check every 2 seconds
    
    # Event handlers
    start_btn.click(
        fn=start_webcam,
        outputs=webcam_output
    )
    
    stop_btn.click(
        fn=stop_webcam,
        outputs=webcam_output
    )
    
    webcam_timer.tick(
        fn=get_webcam_frame,
        outputs=webcam_output,
        show_progress=False
    )
    
    # Conversation control handlers
    start_conv_btn.click(
        fn=start_conversation,
        outputs=status_text
    )
    
    pause_conv_btn.click(
        fn=pause_conversation,
        outputs=status_text
    )
    
    end_conv_btn.click(
        fn=end_conversation,
        outputs=status_text
    )
    
    clear_btn.click(
        fn=lambda: ([], "⏸️ Chat cleared"),
        outputs=[chatbot, status_text]
    )
    
    # Process audio when timer ticks
    conversation_timer.tick(
        fn=process_audio_and_chat,
        inputs=[chatbot],
        outputs=[chatbot, audio_output, status_text],
        show_progress=False
    )

## Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )