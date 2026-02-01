from pathlib import Path
from llama_cpp import Llama
from . import log

logger = log.get_logger()

class LlamaCppTranslator:
    """Translates text using a local GGUF model via llama-cpp-python."""

    def __init__(self, model_path: str, target_language: str = "Brazilian Portuguese"):
        self.model_path = model_path
        self.target_language = target_language
        self._llm = None

    def load(self) -> None:
        """Load the GGUF model."""
        if self._llm is not None:
            return

        if not self.model_path or not Path(self.model_path).exists():
            raise ValueError(f"Model path not found: {self.model_path}")

        logger.info("loading llamacpp model", path=self.model_path)
        
        # Load model with GPU support (n_gpu_layers=-1 for all layers)
        # The user specifically asked for CUDA support for their RTX 5070
        try:
             self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=-1, # Offload all layers to GPU
                n_ctx=2048,      # Context window
                verbose=False    # Reduce C++ logging
            )
             logger.info("llamacpp model loaded")
        except Exception as e:
            logger.error("failed to load llamacpp model", error=str(e))
            raise e

    def translate(self, text: str) -> str:
        """Translate text using the loaded model."""
        if not self._llm:
            self.load()

        # Gemma-2 fix: Merge system instructions into the user prompt
        # instead of using a separate "system" role.
        instruction = (
            f"You are a honest professional translator that doesn't try to change the meaning of a sentence. Translate the following Japanese text to {self.target_language} (Brazil). "
            "Use a natural tone suitable for a visual novel and games. Keep honorifics if necessary."
            "Output only the translated text, do not add any notes or explanations.\n\n"
            f"Japanese: {text}\n"
            "Translation:"
        )

        messages = [
            {"role": "user", "content": instruction}
        ]
        
        logger.info("llamacpp translating", text=text)

        try:
            output = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.3, # Adicionado para mais precisão
                stop=["\n", "Japanese:", "Translation:"],
            )
            
            result = output['choices'][0]['message']['content'].strip()
            logger.info("llamacpp result", result=result)
            return result
        except Exception as e:
            logger.error("llamacpp generation failed", error=str(e))
            return ""

    def is_loaded(self) -> bool:
        return self._llm is not None
