"""
Configuracion central para el sistema de transcripcion y analisis de audios
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas base
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audios" / "MOVIL"
OUTPUT_DIR = BASE_DIR / "output"
TEMPORAL_DIR = OUTPUT_DIR / "audioPro"
TRANSCRIPTION_DIR = OUTPUT_DIR / "transcriptions"
GREETING_DIR = OUTPUT_DIR / "greeting_analysis"
PROMPT_DIR = BASE_DIR / "prompt"

# Archivo de criterios JSON
CRITERIOS_JSON = PROMPT_DIR / "indicaciones_gestion_requerimiento.json"
QUALITY_PROMPTS_JSON = PROMPT_DIR / "quality_agent_prompts.json"
WHISPER_PROMPT_PATH = PROMPT_DIR / "whisper_initial_prompt.txt"
WHISPER_HOTWORDS_PATH = PROMPT_DIR / "whisper_hotwords_context.txt"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "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")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
GPT_MODEL = os.getenv("GPT_MODEL", "qwen/qwen3-vl-8b")

# Configuracion de modelos
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")  # tiny, base, small, medium, large

#lmstudio-community/Meta-Llama-3-8B-Instruct
#dolphin-2.9.3-mistral-nemo-12b-llamacppfixed
# Configuracion de audio
SAMPLE_RATE = 16000  # Hz
MIN_SILENCE_LEN = 1000  # ms - duracion minima de silencio para considerar
SILENCE_THRESH = -40  # dB - umbral de silencio
HOLD_TONE_FREQ_RANGE = (300, 3400)  # Hz - rango de frecuencias de tonos de espera

# Configuracion de diarizacion
MIN_SPEAKERS = 2
MAX_SPEAKERS = 2

# Configuracion de analisis
GREETING_TIMEOUT = 60  # segundos - tiempo maximo para detectar saludo inicial
MIN_CONFIDENCE_SCORE = 0.7  # umbral minimo de confianza para transcripcion

# Palabras clave para identificar al agente (Asesor)
AGENT_KEYWORDS = {
    "le atiende": 5,
    "mi nombre es": 3,
    "en qué puedo ayudarle": 4,
    "buenas tardes": 1,
    "buenos días": 1,
    "me brinda por favor su": 3,
    "asesor": 3,
    "del área de": 3,
    "para servirle": 3,
    "con gusto": 2,
    "su número de documento": 2,
    # Apertura y Presentación
    "bienvenido a": 4,
    "se comunica con": 4,
    "le habla": 3,
    "mi nombre es": 3,
    "gracias por llamar": 4,
    
    # Intención de ayuda
    "en qué puedo ayudarle": 4,
    "en qué puedo asistirle": 5,
    "cómo puedo ayudar": 3,
    "motivo de su llamada": 5,
    
    # Validación y Seguridad (Muy alta probabilidad de ser agente)
    "titular de la línea": 5,
    "validar sus datos": 4,
    "verificar sus datos": 4,
    "por motivos de seguridad": 5,
    "número de documento": 3,
    "número de cédula": 3,
    "me confirma su": 2,
    
    # Manejo de la llamada
    "permítame un momento": 4,
    "gracias por la espera": 4,
    "no retire": 3,
    "sistema está un poco lento": 5,
    "lo transfiero": 4,
    "área de soporte": 4,
    "área de ventas": 4,
    
    # Cierre y Despedida
    "alguna otra duda": 5,
    "alguna otra consulta": 5,
    "número de reporte": 5,
    "número de ticket": 5,
    "encuesta de satisfacción": 5,
    "calificar mi servicio": 5,
    "para servirle": 3,
    "un gusto atenderle": 3,

    # Marcas (Opcional, según tu caso)
    "atención al cliente": 4,
    "me abriría un teléfono móvil diferente": 4
}

# Crear directorios si no existen
for directory in [AUDIO_DIR, TRANSCRIPTION_DIR, GREETING_DIR, PROMPT_DIR, TEMPORAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
