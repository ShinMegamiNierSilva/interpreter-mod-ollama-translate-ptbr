# Interpreter (Ollama Mod)

An enhanced version of **Interpreter**, an offline screen translator for Japanese retro games. This modification adds support for **Ollama** and **LlamaCPP** (CUDA), allowing you to translate Japanese game text into **Thai**, English, or any other language supported by modern LLMs.

![screenshot](screenshot.png)

## New Features (Mod)

-   **Ollama Support**: Use local LLMs via Ollama (e.g., `gemma3:4b`) for high-quality translation.
-   **LlamaCPP (CUDA)**: Run GGUF models directly with GPU acceleration for maximum speed.
-   **CUDA Powered OCR**: Upgraded `MeikiOCR` to use `onnxruntime-gpu` for faster text detection.
-   **Thai Language Support**: Translate Japanese directly to Thai (or any target language).
-   **Translation Caching**: Intelligent caching key-value store to stabilize LLM outputs and prevent flickering.
-   **Enhanced GUI**: New settings for backend selection (Sugoi / Ollama / LlamaCPP) and model configuration.

## Original Features

-   **Fully offline** - No cloud APIs, no internet required after setup
-   **Optimized for retro games** - Uses MeikiOCR, trained specifically on Japanese game text
-   **Two overlay modes** - Banner (subtitle bar) or inplace (text over game)
-   **Multi-display support** - Overlay appears on the same display as the game

## Requirements

-   **Windows 10/11** (Tested on Windows 11)
-   **NVIDIA GPU** (Tested with RTX 5070) - Required for CUDA acceleration
-   **Python 3.12+** (Required for `llama-cpp-python` wheel)
-   **Ollama** (Optional, for Ollama backend)

## Installation

1.  **Install `uv`** (Universal Python Package Manager):
    ```powershell
    pip install uv
    ```

2.  **Clone the repository**:
    ```powershell
    git clone https://github.com/quutamo888/interpreter-mod-ollama-translate-thai
    cd interpreter-mod-ollama-translate-thai
    ```

3.  **Install Custom LlamaCPP Wheel** (Required for CUDA support):
    This repository includes a custom-built wheel for `llama-cpp-python` with CUDA support for Python 3.12.
    ```powershell
    uv add "llama_cpp_python-0.3.16-cp312-cp312-win_amd64.whl"
    ```

4.  **Run the application**:
    ```powershell
    uv run interpreter-v2
    ```
    (Other dependencies will be installed automatically on the first run).

## Configuration

### Translation Backends

Go to the **Translation** tab in the GUI to switch backends:

1.  **Sugoi V4 (Offline)**:
    -   Original backend. Fast, Japanese -> English only.
    -   Fully offline, no extra setup.

2.  **Ollama (Local LLM)**:
    -   Requires [Ollama](https://ollama.com/) installed and running (`ollama serve`).
    -   **Model**: e.g., `gemma3:4b` (Pull it first: `ollama pull gemma3:4b`).
    -   **Target Language**: e.g., `Thai`.

3.  **LlamaCPP (CUDA/GGUF)**:
    -   Runs `.gguf` models directly with CUDA acceleration.
    -   **Model Path**: Browse to your `.gguf` file (e.g., `HY-MT1.5-1.8B-Q8_0.gguf`).
    -   **Target Language**: e.g., `Thai`.
    -   **Note**: Models should be "Instruction Tuned" (e.g., `gemma-it`) for best results.

## How It Works

1.  **Capture**: Captures the game window.
2.  **OCR**: Extracts Japanese text using **MeikiOCR** (GPU accelerated).
3.  **Translation**: 
    -   Checks **Translation Cache** (fuzzy match) to see if we already translated this text (prevents flickering).
    -   If new, sends to **Ollama** or **LlamaCPP** for translation to Thai.
4.  **Overlay**: Displays the result.

## Troubleshooting

### Capture button "doesn't work"
-   Ensure you have an NVIDIA GPU and CUDA drivers installed.
-   The app tries to load CUDA 12 binaries.

### Translation "flickers"
-   This is normal for LLMs, but the added **Translation Cache** significantly reduces this by reusing translations for static text.

### Empty Translation Output
-   Ensure your model supports the target language.
-   For LlamaCPP, ensure you are using an Instruction-tuned model (`-it` or `-instruct`) compatible with the internal chat template.

