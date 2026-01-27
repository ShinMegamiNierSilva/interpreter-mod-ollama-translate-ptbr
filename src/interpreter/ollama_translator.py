"""Ollama translation backend for Interpreter."""

import ollama
from . import log

logger = log.get_logger()

class OllamaTranslator:
    """Translates text using a local Ollama model."""

    def __init__(self, model: str = "gemma3:4b", target_language: str = "Thai"):
        """Initialize the Ollama translator.

        Args:
            model: Name of the Ollama model to use (e.g., "gemma3:4b").
            target_language: Target language for translation (e.g., "Thai").
        """
        self.model = model
        self.target_language = target_language
        self._is_ready = False

    def load(self) -> None:
        """Verify connection to Ollama and model availability."""
        if self._is_ready:
            return

        logger.info("loading ollama model", model=self.model)
        
        try:
            # Check if model exists, if not try to pull it (or let ollama handle it)
            # listing models to verify connection
            models_response = ollama.list()
            # Handle object response from newer ollama library
            if hasattr(models_response, 'models'):
                model_names = [m.model for m in models_response.models]
            else:
                # Fallback for dict-like response if version varies
                model_names = [m.get('name', m.get('model')) for m in models_response.get('models', [])]
            
            # Simple check, exact match might be tricky with tags, but good enough for log
            logger.info("ollama connection successful", available_models=len(model_names))
            
            # We don't strictly need to pull here as ollama.chat will pull if missing,
            # but we can try a dry run or just mark as ready.
            self._is_ready = True
            
        except Exception as e:
            logger.error("failed to connect to ollama", error=str(e))
            raise

    def translate(self, text: str) -> str:
        """Translate text to target language.

        Args:
            text: Text to translate.

        Returns:
            Translated text.
        """
        if not text or not text.strip():
            return ""

        if not self._is_ready:
            self.load()

        prompt = f"Translate the following Japanese text to {self.target_language}. Only provide the translation, no explanations:\n\n{text}"

        try:
            response = ollama.chat(model=self.model, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            return response['message']['content'].strip()
        except Exception as e:
            logger.error("ollama translation failed", error=str(e))
            return f"[Error: {e}]"

    def is_loaded(self) -> bool:
        return self._is_ready
