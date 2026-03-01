# 📻 Tactical Radio Intercept Analyzer

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/) 
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green)](https://python.langchain.com/)

##  Overview
This repository contains an R&D project focused on the automated classification of tactical radio intercepts. The system utilizes a multi-modal AI pipeline to transcribe noisy, real-world radio communications and classify them into structured data using Large Language Models (LLMs). 

Developed as a complex academic research project, this tool demonstrates the practical application of NLP and Audio Processing in military/tactical contexts.

## Architecture & Pipeline
The system operates in two main stages:
1. **Automatic Speech Recognition (ASR):** Uses OpenAI's `whisper-large-v3-turbo` (via Hugging Face `transformers`) to transcribe Ukrainian audio, specifically optimized to handle severe static, background noise, and clipping typical of tactical comms.
2. **LLM Text Classification:** Employs Google's `gemini-2.5-flash` via `LangChain`. A strict system prompt and `Pydantic` output parsers force the model to categorize the messy transcription into predefined tactical clusters (e.g., Reconnaissance, Medevac, Artillery, Logistics) and extract entities like coordinates and callsigns into a clean JSON format.

## Features
* **End-to-End Processing:** From raw `.wav`/`.ogg` audio straight to a structured JSON intelligence report.
* **Resilient ASR:** Capable of understanding heavily distorted speech (simulated radio chatter).
* **Hallucination Control:** Zero-temperature LLM generation with strict schema enforcement.
* **Interactive UI:** Built-in Gradio web interface for live audio recording and file uploads.
* **DSP Dataset Generator:** Includes a custom Digital Signal Processing script (`batch_radio_fx.py`) to apply bandpass filters, overdrive, and white noise to clean audio, simulating gritty tactical environments.

## Quick Start (Local Setup)

### Prerequisites
* Python 3.10+
* FFmpeg installed on your system.
* Google Gemini API Key.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com AlexanderSychev2005/radio-analysis.git
   cd radio-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your API Key:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

## Running the app
Launch the interactive web interface:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

## Dataset Generation (Radio FX)
To test the model, you can generate your own tactical intercepts. Place clean .wav or .ogg voice recordings in the test_audios/ folder and run:
```bash
python radio_fx.py
```
This script applies mathematical filters (cutting frequencies below 300Hz and above 3000Hz, applying clipping, and adding white noise) and outputs the results to the radio_audios/ folder.

##  Model Metrics & Evaluation

The ASR component is evaluated based on the Word Error Rate (WER). While the model may produce slight transcription errors due to induced radio static, the subsequent LLM's self-attention mechanism effectively contextualizes and corrects these discrepancies during the classification stage.

---
*Developed for research and educational purposes.*