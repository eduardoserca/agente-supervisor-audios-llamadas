"""
Módulo para calcular métricas de transcripción
"""
from typing import Dict, List
import numpy as np


class MetricsCalculator:
    """Calculador de métricas para transcripciones"""
    
    def __init__(self):
        pass
    
    def calculate_speaker_metrics(self, segments: List[Dict]) -> Dict:
        """
        Calcula métricas por speaker
        
        Args:
            segments: Lista de segmentos con información de speaker y texto
                     Formato: [{"speaker": "SPEAKER_00", "text": "...", "start": 0.0, "end": 5.0}, ...]
        
        Returns:
            Diccionario con métricas por speaker
        """
        speaker_stats = {}
        
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            role = segment.get("role", "UNKNOWN")
            text = segment.get("text", "")
            duration = segment.get("end", 0) - segment.get("start", 0)
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "role": role,
                    "total_duration": 0.0,
                    "word_count": 0,
                    "segment_count": 0,
                    "texts": []
                }
            
            speaker_stats[speaker]["total_duration"] += duration
            speaker_stats[speaker]["word_count"] += len(text.split())
            speaker_stats[speaker]["segment_count"] += 1
            speaker_stats[speaker]["texts"].append(text)
        
        # Calcular porcentajes
        total_duration = sum(stats["total_duration"] for stats in speaker_stats.values())
        
        for speaker in speaker_stats:
            speaker_stats[speaker]["participation_percentage"] = (
                speaker_stats[speaker]["total_duration"] / total_duration * 100
                if total_duration > 0 else 0
            )
            speaker_stats[speaker]["average_words_per_segment"] = (
                speaker_stats[speaker]["word_count"] / speaker_stats[speaker]["segment_count"]
                if speaker_stats[speaker]["segment_count"] > 0 else 0
            )
        
        return speaker_stats
    
        return speaker_stats
    
    def calculate_role_metrics(self, segments: List[Dict]) -> Dict:
        """
        Calcula métricas por rol (AGENT vs CLIENT)
        """
        role_stats = {
            "AGENT": {"word_count": 0, "duration": 0.0, "segments": 0},
            "CLIENT": {"word_count": 0, "duration": 0.0, "segments": 0},
            "UNKNOWN": {"word_count": 0, "duration": 0.0, "segments": 0}
        }
        
        for seg in segments:
            role = seg.get("role", "UNKNOWN")
            if role not in role_stats:
                role = "UNKNOWN"
            
            duration = seg.get("end", 0) - seg.get("start", 0)
            text_len = len(seg.get("text", "").split())
            
            role_stats[role]["duration"] += duration
            role_stats[role]["word_count"] += text_len
            role_stats[role]["segments"] += 1
            
        # Calcular porcentajes de habla
        total_duration = sum(s["duration"] for s in role_stats.values())
        if total_duration > 0:
            for role in role_stats:
                role_stats[role]["talk_percentage"] = (role_stats[role]["duration"] / total_duration) * 100
        
        return role_stats

    def calculate_transcription_confidence(self, segments: List[Dict]) -> Dict:
        """
        Calcula métricas de confianza de la transcripción
        """
        if not segments:
            return {
                "average_confidence": 0.0,
                "average_logprob": 0.0,
                "low_confidence_segments": 0
            }
        
        confidences = [seg.get("confidence", 0.8) for seg in segments]
        logprobs = [seg.get("avg_logprob", -0.5) for seg in segments]
        
        # Normalizar logprob a un score 0-1 (típicamente -2.0 a -0.0)
        # Un valor de -1.0 o superior es decente. -0.2 es excelente.
        logprob_scores = [min(1.0, max(0.0, (lp + 2.0) / 2.0)) for lp in logprobs]
        
        return {
            "average_confidence": float(np.mean(confidences)),
            "average_logprob": float(np.mean(logprobs)),
            "average_logprob_score": float(np.mean(logprob_scores)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
            "low_confidence_segments": sum(1 for c in confidences if c < 0.7)
        }
    
    def calculate_overall_metrics(self, segments: List[Dict], 
                                  processing_time: float,
                                  audio_duration: float) -> Dict:
        """
        Calcula métricas generales de la transcripción
        """
        full_text = " ".join(seg.get("text", "") for seg in segments)
        words = full_text.split()
        
        return {
            "total_segments": len(segments),
            "total_words": len(words),
            "total_characters": len(full_text),
            "audio_duration_seconds": audio_duration,
            "processing_time_seconds": processing_time,
            "processing_speed_ratio": audio_duration / processing_time if processing_time > 0 else 0,
            "words_per_minute": (len(words) / audio_duration * 60) if audio_duration > 0 else 0,
            "average_segment_duration": audio_duration / len(segments) if segments else 0
        }
    
    def generate_full_report(self, segments: List[Dict],
                           processing_time: float,
                           audio_duration: float,
                           preprocessing_metrics: Dict = None) -> Dict:
        """
        Genera un reporte completo de métricas mejorado
        """
        report = {
            "overall_metrics": self.calculate_overall_metrics(
                segments, processing_time, audio_duration
            ),
            "speaker_metrics": self.calculate_speaker_metrics(segments),
            "role_metrics": self.calculate_role_metrics(segments),
            "confidence_metrics": self.calculate_transcription_confidence(segments)
        }
        
        if preprocessing_metrics:
            report["preprocessing_metrics"] = preprocessing_metrics
        
        # --- CÁLCULO MEJORADO DE QUALITY SCORE ---
        # Combinamos: 
        # 1. Confianza base (probabilidad de que sea voz y no ruido)
        # 2. Logprob score (calidad de "entendimiento" del modelo)
        # 3. Penalización por fragmentos de baja confianza
        
        conf_metrics = report["confidence_metrics"]
        base_confidence = conf_metrics["average_confidence"]
        logprob_score = conf_metrics.get("average_logprob_score", base_confidence)
        
        low_conf_ratio = (
            conf_metrics["low_confidence_segments"] / 
            len(segments) if segments else 0
        )
        
        # Ponderación: 70% logprob (entendimiento), 30% confianza (detección voz)
        weighted_quality = (logprob_score * 0.7) + (base_confidence * 0.3)
        
        # Penalización exponencial suave por segmentos malos
        final_quality = weighted_quality * (1.0 - (low_conf_ratio ** 2) * 0.5)
        
        report["quality_score"] = round(min(100, final_quality * 100), 2)
        
        return report
