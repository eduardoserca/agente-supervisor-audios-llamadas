# Sistema de Transcripción y Análisis de Audios

Sistema completo para transcribir audios a texto con identificación de participantes y análisis de cumplimiento de criterios de calidad en atención al cliente.

## 🎯 Características

### Agente 1: Transcripción de Audio
- **Transcripción automática** usando Whisper (OpenAI)
- **Identificación de participantes** (speaker diarization) con pyannote.audio
- **Eliminación de silencios** y tonos de espera
- **Métricas detalladas** de calidad y confianza
- **Reportes JSON** por cada audio procesado

### Agente 2: Análisis de Saludos
- **Análisis inteligente** de saludos usando IA (GPT) o reglas
- **Validación de cumplimiento** según criterios JSON
- **Detección de elementos clave**: saludo, identificación, ofrecimiento de ayuda
- **Reportes de compliance** con recomendaciones

### Notebook Interactivo
- **Ejecución paso a paso** de ambos agentes
- **Visualizaciones interactivas** con Plotly
- **Dashboard consolidado** con métricas clave
- **Exportación de reportes** en CSV y HTML

## 📋 Requisitos Previos

- Python 3.8 o superior
- CUDA (opcional, para GPU acceleration)
- Claves API:
  - OpenAI API Key (para Whisper y análisis GPT)
  - Hugging Face Token (para speaker diarization)

## 🚀 Instalación

### 1. Clonar o descargar el proyecto

```bash
cd d:\Proy\AntiG\MIA\NLP\TrabajoFinal
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
venv\Scripts\activate  # En Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Copiar el archivo `.env.example` a `.env` y completar con tus claves:

```bash
copy .env.example .env
```

Editar `.env`:
```
OPENAI_API_KEY=tu_clave_openai_aqui
HUGGINGFACE_TOKEN=tu_token_huggingface_aqui
```

Para obtener las claves:
- **OpenAI**: https://platform.openai.com/api-keys
- **Hugging Face**: https://huggingface.co/settings/tokens

## 📁 Estructura del Proyecto

```
TrabajoFinal/
├── agents/
│   ├── transcription_agent.py      # Agente de transcripción
│   ├── greeting_agent.py           # Agente de análisis de saludos
│   ├── audio_processor.py          # Procesamiento de audio
│   ├── metrics_calculator.py       # Cálculo de métricas
│   └── prompt_loader.py            # Cargador de criterios JSON
├── data/
│   └── audios/                     # Coloca aquí tus archivos de audio
├── output/
│   ├── transcriptions/             # Transcripciones generadas
│   └── greeting_analysis/          # Análisis de saludos
├── pront/                          # Prompts adicionales (opcional)
├── analisis_audios.ipynb           # Notebook principal
├── config.py                       # Configuración del sistema
├── requirements.txt                # Dependencias
├── .env.example                    # Plantilla de variables de entorno
├── indicaciones_gestion_requerimiento.json  # Criterios de evaluación
└── README.md                       # Este archivo
```

## 💻 Uso

### Opción 1: Usando el Notebook (Recomendado)

1. Colocar archivos de audio en `data/audios/`
2. Abrir Jupyter:
   ```bash
   jupyter notebook analisis_audios.ipynb
   ```
3. Ejecutar las celdas secuencialmente
4. Ver resultados y visualizaciones interactivas

### Opción 2: Usando los Agentes Directamente

#### Transcripción:

```python
from agents.transcription_agent import TranscriptionAgent

# Inicializar agente
agent = TranscriptionAgent(whisper_model="medium")

# Procesar todos los audios
results = agent.process_directory()

# O procesar un audio específico
result = agent.process_audio("data/audios/mi_audio.wav")
agent.save_transcription(result, "output/transcriptions/resultado.json")
```

#### Análisis de Saludos:

```python
from agents.greeting_agent import GreetingAgent

# Inicializar agente
agent = GreetingAgent(use_ai=True)

# Procesar todas las transcripciones
results = agent.process_directory()

# O procesar una transcripción específica
result = agent.process_transcription_file(
    "output/transcriptions/mi_audio_transcription.json"
)
```

## 📊 Métricas Generadas

### Métricas de Transcripción
- **Calidad general** (0-100%)
- **Confianza promedio** de la transcripción
- **Distribución por speaker** (tiempo, palabras, participación)
- **Estadísticas de procesamiento** (tiempo, velocidad)
- **Métricas de preprocesamiento** (silencio removido, tonos detectados)

### Métricas de Saludos
- **Puntuación de empatía** (0-100)
- **Cumplimiento** (CUMPLE/NO CUMPLE)
- **Elementos detectados**: saludo, identificación, ofrecimiento
- **Recomendaciones** de mejora

## 🎨 Visualizaciones

El notebook genera:
- Gráfico de barras de calidad por audio
- Distribución de participación por speaker
- Puntuación de saludos con umbral de cumplimiento
- Radar chart de elementos del saludo
- Dashboard consolidado con múltiples métricas

## ⚙️ Configuración Avanzada

Editar `config.py` para ajustar:

```python
# Modelo de Whisper (tiny, base, small, medium, large)
WHISPER_MODEL = "medium"

# Modelo GPT para análisis
GPT_MODEL = "gpt-4-turbo-preview"

# Parámetros de audio
SAMPLE_RATE = 16000
MIN_SILENCE_LEN = 1000  # ms
SILENCE_THRESH = -40    # dB

# Diarización
MIN_SPEAKERS = 2
MAX_SPEAKERS = 5

# Análisis de saludos
GREETING_TIMEOUT = 60   # segundos
```

## 🔧 Solución de Problemas

### Error: "No module named 'whisper'"
```bash
pip install openai-whisper
```

### Error: "CUDA out of memory"
- Usar modelo Whisper más pequeño: `WHISPER_MODEL = "small"`
- O desactivar GPU: `use_gpu=False`

### Error: "Hugging Face authentication"
- Verificar token en `.env`
- Aceptar términos en: https://huggingface.co/pyannote/speaker-diarization-3.1

### No se encuentran archivos de audio
- Verificar que los archivos estén en `data/audios/`
- Formatos soportados: WAV, MP3, M4A, FLAC, OGG

## 📝 Formatos de Salida

### Transcripción (JSON)
```json
{
  "audio_file": "ejemplo.wav",
  "timestamp": "2025-11-29T14:00:00",
  "full_text": "Transcripción completa...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "speaker": "SPEAKER_00",
      "text": "Hola, buenos días...",
      "confidence": 0.95
    }
  ],
  "metrics": {
    "quality_score": 92.5,
    "speaker_metrics": {...},
    "confidence_metrics": {...}
  }
}
```

### Análisis de Saludos (JSON)
```json
{
  "audio_file": "ejemplo.wav",
  "compliance": {
    "R2_empatia_claridad": "CUMPLE",
    "score": 85
  },
  "greeting_analysis": {
    "cumple_saludo": true,
    "tiene_identificacion": true,
    "puntuacion_empatia": 85,
    "elementos_positivos": [...],
    "elementos_mejora": [...]
  }
}
```

## 🤝 Contribuciones

Este proyecto fue desarrollado para análisis de calidad en atención al cliente.

## 📄 Licencia

Proyecto académico - NLP - Trabajo Final

## 📧 Soporte

Para problemas o preguntas, revisar:
1. Este README
2. Comentarios en el código
3. Documentación de las librerías utilizadas

---

**Desarrollado con ❤️ para análisis de calidad en servicio al cliente**
