"""
Paquete de agentes para transcripcion y analisis de audios
"""

__version__ = "1.0.0"
__author__ = "Audio Analysis System"

from .transcription_agent import TranscriptionAgent
from .greeting_agent import GreetingAgent
from .audio_processor import AudioProcessor
from .metrics_calculator import MetricsCalculator
from .prompt_loader import PromptLoader

__all__ = [
    'TranscriptionAgent',
    'GreetingAgent',
    'AudioProcessor',
    'MetricsCalculator',
    'PromptLoader'
]
