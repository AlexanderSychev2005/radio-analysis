import torch
from transformers import pipeline


def setup_transcriber(model: str = "openai/whisper-large-v3-turbo"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Model: {model} loading on {device} device")
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device,
        torch_dtype=torch_dtype,
    )
    return transcriber


def process_audio(transcriber, audio_path: str):
    """ """
    print(f"Radion communication analysis {audio_path}")

    result = transcriber(audio_path, generate_kwargs={"language": "ukrainian"})
    return result["text"].strip()


if __name__ == "__main__":
    asr_pipeline = setup_transcriber("openai/whisper-large-v3-turbo")
    test_audio = "radio_intercept_01.wav"
