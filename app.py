import gradio as gr
import json
from main import TacticalRadioAnalyzer

print("Starting server and loading models...")
analyzer = TacticalRadioAnalyzer()


def process_audio_for_ui(audio_filepath):
    """
    Wrapper function that takes audio from Gradio and returns results.
    """
    if not audio_filepath:
        gr.Warning("Please upload or record an audio file first.")
        return "Error: No audio provided.", "{}"

    # Call the main processing pipeline
    result = analyzer.process_intercept(audio_filepath)

    if result.get("status") == "success":
        # If successful, extract transcription and format JSON for clean UI display
        transcription = result["transcription"]
        analysis_json = json.dumps(result["analysis"], indent=4, ensure_ascii=False)
        return transcription, analysis_json
    else:
        # Error handling: extract step and message
        error_msg = result.get("message", "Unknown error occurred.")
        step = result.get("step", "Unknown step")

        # Use fallback transcription if ASR passed but LLM failed
        transcription = result.get("transcription", f"Failed during {step} stage.")

        error_report = {"error": True, "failed_step": step, "details": error_msg}
        error_json = json.dumps(error_report, indent=4, ensure_ascii=False)

        gr.Error(f"Processing failed at {step} stage. See JSON for details.")
        return transcription, error_json


with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 📻 Tactical Radio Intercept Analyzer")
    gr.Markdown(
        "**R&D Project | Textual Information Classification using ASR and LLMs**"
    )

    with gr.Row():
        # Left Column: Input Data
        with gr.Column():
            gr.Markdown("### 1. Data Source")
            # Audio widget allows both file upload and microphone recording
            audio_input = gr.Audio(
                type="filepath", label="Upload audio file or record voice"
            )
            analyze_btn = gr.Button("⚡ Analyze Intercept", variant="primary")

        # Right Column: Results
        with gr.Column():
            gr.Markdown("### 2. Analysis Results")
            transcription_output = gr.Textbox(
                label="Transcribed Text (Whisper)", lines=3
            )
            json_output = gr.Code(
                label="Structured Report (Gemini API)", language="json"
            )

    # Link the button click to the processing function
    analyze_btn.click(
        fn=process_audio_for_ui,
        inputs=audio_input,
        outputs=[transcription_output, json_output],
    )

if __name__ == "__main__":
    # Launch the web server (standard launch for HF Spaces)
    demo.launch()
