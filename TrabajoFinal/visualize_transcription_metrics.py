"""
Modulo para visualizar metricas de transcripcion de audio a texto
Genera visualizaciones interactivas con Plotly
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import config


def load_transcription_metrics(transcription_dir: str = None) -> pd.DataFrame:
    """
    Carga metricas de todos los archivos de transcripcion
    
    Args:
        transcription_dir: Directorio con archivos de transcripcion JSON
    
    Returns:
        DataFrame con metricas consolidadas
    """
    transcription_dir = Path(transcription_dir or config.TRANSCRIPTION_DIR)
    
    # Buscar archivos de transcripcion
    transcription_files = list(transcription_dir.glob("*_transcription.json"))
    
    if not transcription_files:
        print(f"No se encontraron archivos de transcripcion en {transcription_dir}")
        return pd.DataFrame()
    
    print(f"Cargando metricas de {len(transcription_files)} transcripciones...")
    
    data = []
    for trans_file in transcription_files:
        try:
            with open(trans_file, 'r', encoding='utf-8') as f:
                trans_data = json.load(f)
            
            metrics = trans_data.get('metrics', {})
            overall = metrics.get('overall_metrics', {})
            preprocessing = metrics.get('preprocessing_metrics', {})
            speaker_metrics = metrics.get('speaker_metrics', {})
            segments = trans_data.get('segments', [])
            
            # Calcular confianza promedio
            confidences = [seg.get('confidence', 0) for seg in segments if 'confidence' in seg]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Contar speakers unicos
            speakers = set(seg.get('speaker', 'UNKNOWN') for seg in segments)
            num_speakers = len(speakers)
            
            # Calcular tiempo por speaker
            speaker_times = {}
            for speaker, speaker_data in speaker_metrics.items():
                speaker_times[speaker] = speaker_data.get('total_speaking_time', 0)
            
            row = {
                'archivo': trans_data.get('audio_file', trans_file.stem),
                'quality_score': metrics.get('quality_score', 0),
                'audio_duration': overall.get('audio_duration_seconds', 0),
                'processing_time': overall.get('processing_time_seconds', 0),
                'avg_confidence': avg_confidence,
                'num_speakers': num_speakers,
                'num_segments': len(segments),
                'silence_removed_pct': preprocessing.get('silence_removed_percentage', 0),
                'hold_tones': preprocessing.get('hold_tones_detected', 0),
                'original_duration': preprocessing.get('original_duration', 0),
                'processed_duration': preprocessing.get('processed_duration', 0),
                'speaker_times': speaker_times,
                'full_text_length': len(trans_data.get('full_text', ''))
            }
            
            # Calcular ratio de tiempo real
            if row['audio_duration'] > 0:
                row['realtime_ratio'] = row['processing_time'] / row['audio_duration']
            else:
                row['realtime_ratio'] = 0
            
            # Calcular reduccion de duracion
            if row['original_duration'] > 0:
                row['duration_reduction_pct'] = ((row['original_duration'] - row['processed_duration']) / 
                                                 row['original_duration'] * 100)
            else:
                row['duration_reduction_pct'] = 0
            
            data.append(row)
            
        except Exception as e:
            print(f"Error cargando {trans_file.name}: {e}")
    
    df = pd.DataFrame(data)
    print(f"Cargadas {len(df)} transcripciones exitosamente")
    
    return df


def plot_quality_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Visualiza la distribucion de quality scores
    
    Args:
        df: DataFrame con metricas
    
    Returns:
        Figura de Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribucion de Quality Scores', 'Box Plot de Calidad'),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )
    
    # Histograma
    fig.add_trace(
        go.Histogram(
            x=df['quality_score'],
            nbinsx=20,
            name='Quality Score',
            marker_color='rgb(55, 83, 109)',
            hovertemplate='Score: %{x}<br>Frecuencia: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(
            y=df['quality_score'],
            name='Quality Score',
            marker_color='rgb(26, 118, 255)',
            boxmean='sd',
            hovertemplate='Score: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Quality Score", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
    fig.update_yaxes(title_text="Quality Score", row=1, col=2)
    
    fig.update_layout(
        title_text="Analisis de Calidad de Transcripcion",
        showlegend=False,
        height=400,
        hovermode='closest'
    )
    
    return fig


def plot_confidence_scores(df: pd.DataFrame) -> go.Figure:
    """
    Visualiza scores de confianza por archivo
    
    Args:
        df: DataFrame con metricas
    
    Returns:
        Figura de Plotly
    """
    # Ordenar por confianza
    df_sorted = df.sort_values('quality_score', ascending=True)
    
    # Crear colores basados en confianza
    colors = ['red' if x < 70 else 'orange' if x < 85 else 'green' 
              for x in df_sorted['quality_score']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df_sorted['archivo'],
            x=df_sorted['quality_score'],
            orientation='h',
            marker=dict(color=colors),
            hovertemplate='<b>%{y}</b><br>Confianza: %{x}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Confianza Promedio por Transcripcion",
        xaxis_title="Confianza Promedio",
        yaxis_title="Archivo",
        height=max(400, len(df) * 20),
        hovermode='closest'
        #xaxis=dict(range=[0.6, 1])
    )
    
    # Agregar lineas de referencia
    fig.add_vline(x=70, line_dash="dash", line_color="red", 
                  annotation_text="Baja", annotation_position="top")
    fig.add_vline(x=85, line_dash="dash", line_color="orange",
                  annotation_text="Media", annotation_position="top")
    
    return fig


def plot_duration_vs_quality(df: pd.DataFrame) -> go.Figure:
    """
    Visualiza relacion entre duracion y calidad
    
    Args:
        df: DataFrame con metricas
    
    Returns:
        Figura de Plotly
    """
    fig = px.scatter(
        df,
        x='audio_duration',
        y='quality_score',
        size='num_segments',
        color='avg_confidence',
        hover_data=['archivo', 'num_speakers'],
        labels={
            'audio_duration': 'Duracion del Audio (segundos)',
            'quality_score': 'Quality Score',
            'num_segments': 'Numero de Segmentos',
            'avg_confidence': 'Confianza Promedio'
        },
        title="Relacion Duracion vs Calidad",
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      'Duracion: %{x:.1f}s<br>' +
                      'Quality: %{y:.1f}<br>' +
                      'Speakers: %{customdata[1]}<br>' +
                      'Confianza: %{marker.color:.2%}<extra></extra>'
    )
    
    fig.update_layout(height=500, hovermode='closest')
    
    return fig


def plot_processing_efficiency(df: pd.DataFrame) -> go.Figure:
    """
    Visualiza eficiencia de procesamiento
    
    Args:
        df: DataFrame con metricas
    
    Returns:
        Figura de Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Tiempo de Procesamiento', 'Ratio de Tiempo Real'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Ordenar por tiempo de procesamiento
    df_sorted = df.sort_values('processing_time', ascending=False).head(15)
    
    # Tiempo de procesamiento
    fig.add_trace(
        go.Bar(
            x=df_sorted['archivo'],
            y=df_sorted['processing_time'],
            name='Tiempo de Procesamiento',
            marker_color='rgb(55, 83, 109)',
            hovertemplate='<b>%{x}</b><br>Tiempo: %{y:.2f}s<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Ratio de tiempo real
    fig.add_trace(
        go.Bar(
            x=df_sorted['archivo'],
            y=df_sorted['realtime_ratio'],
            name='Ratio Tiempo Real',
            marker_color='rgb(26, 118, 255)',
            hovertemplate='<b>%{x}</b><br>Ratio: %{y:.2f}x<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Archivo", tickangle=-45, row=1, col=1)
    fig.update_xaxes(title_text="Archivo", tickangle=-45, row=1, col=2)
    fig.update_yaxes(title_text="Segundos", row=1, col=1)
    fig.update_yaxes(title_text="Ratio (x)", row=1, col=2)
    
    fig.update_layout(
        title_text="Eficiencia de Procesamiento",
        showlegend=False,
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_preprocessing_effectiveness(df: pd.DataFrame) -> go.Figure:
    """
    Visualiza efectividad del preprocesamiento
    
    Args:
        df: DataFrame con metricas
    
    Returns:
        Figura de Plotly
    """
    fig = make_subplots(
        rows=1, cols=1,
        #subplot_titles=('Duracion: Original vs Procesada'),
        specs=[
            [{"type": "bar"}]
        ]
    )
    
    # Duracion original vs procesada
    df_top = df.sort_values('original_duration', ascending=False)
    fig.add_trace(
        go.Bar(
            y=df_top['archivo'],
            x=df_top['original_duration'],
            orientation='h',
            name='Original',
            marker_color='lightgray',
            hovertemplate='<b>%{x}</b><br>Original: %{y}s<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            y=df_top['archivo'],
            x=df_top['processed_duration'],
            orientation='h',
            name='Procesada',
            marker_color='rgb(44, 160, 44)',
            hovertemplate='<b>%{x}</b><br>Procesada: {y}s<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.update_layout(
        title_text="Duracion: Original vs Procesada",
        showlegend=True,
        #height=800,
        height=max(400, len(df) * 20),
        hovermode='closest'
    )
    
    return fig


def plot_speaker_analysis(df: pd.DataFrame) -> go.Figure:
    """
    Visualiza analisis de speakers
    
    Args:
        df: DataFrame con metricas
    
    Returns:
        Figura de Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribucion de Speakers', 'Tiempo Total por Speaker'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Distribucion de numero de speakers
    speaker_counts = df['num_speakers'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=speaker_counts.index,
            y=speaker_counts.values,
            marker_color='rgb(55, 126, 184)',
            hovertemplate='Speakers: %{x}<br>Archivos: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Tiempo total por speaker (agregado)
    all_speaker_times = {}
    for speaker_times in df['speaker_times']:
        for speaker, time in speaker_times.items():
            all_speaker_times[speaker] = all_speaker_times.get(speaker, 0) + time
    
    if all_speaker_times:
        speakers = list(all_speaker_times.keys())
        times = list(all_speaker_times.values())
        
        fig.add_trace(
            go.Pie(
                labels=speakers,
                values=times,
                hovertemplate='<b>%{label}</b><br>Tiempo: %{value:.1f}s<br>%{percent}<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Numero de Speakers", row=1, col=1)
    fig.update_yaxes(title_text="Numero de Archivos", row=1, col=1)
    
    fig.update_layout(
        title_text="Analisis de Speakers",
        showlegend=True,
        height=400,
        hovermode='closest'
    )
    
    return fig


def generate_summary_dashboard(df: pd.DataFrame) -> go.Figure:
    """
    Genera dashboard con KPIs y resumen
    
    Args:
        df: DataFrame con metricas
    
    Returns:
        Figura de Plotly
    """
    # Calcular KPIs
    avg_quality = df['quality_score'].mean()
    avg_confidence = df['avg_confidence'].mean()
    total_processing_time = df['processing_time'].sum()
    total_audio_duration = df['audio_duration'].sum()
    total_files = len(df)
    
    # Crear figura con indicadores
    fig = go.Figure()
    
    # Agregar indicadores como anotaciones
    fig.add_annotation(
        text=f"<b>Quality Score Promedio</b><br>{avg_quality:.1f}/100",
        xref="paper", yref="paper",
        x=0.15, y=0.9, showarrow=False,
        font=dict(size=16, color="rgb(55, 83, 109)"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgb(55, 83, 109)",
        borderwidth=2,
        borderpad=10
    )
    
    #fig.add_annotation(
    #    text=f"<b>Confianza Promedio</b><br>{avg_confidence:.1%}",
    #    xref="paper", yref="paper",
    #    x=0.5, y=0.9, showarrow=False,
    #    font=dict(size=16, color="rgb(26, 118, 255)"),
    #    bgcolor="rgba(255, 255, 255, 0.8)",
    #    bordercolor="rgb(26, 118, 255)",
    #    borderwidth=2,
    #    borderpad=10
    #)
    
    fig.add_annotation(
        text=f"<b>Archivos Procesados</b><br>{total_files}",
        xref="paper", yref="paper",
        x=0.85, y=0.9, showarrow=False,
        font=dict(size=16, color="rgb(44, 160, 44)"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgb(44, 160, 44)",
        borderwidth=2,
        borderpad=10
    )
    
    fig.add_annotation(
        text=f"<b>Tiempo Total</b><br>{total_processing_time/60:.1f} min",
        xref="paper", yref="paper",
        x=0.15, y=0.6, showarrow=False,
        font=dict(size=16, color="rgb(255, 127, 14)"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgb(255, 127, 14)",
        borderwidth=2,
        borderpad=10
    )
    
    fig.add_annotation(
        text=f"<b>Audio Total</b><br>{total_audio_duration/60:.1f} min",
        xref="paper", yref="paper",
        x=0.5, y=0.6, showarrow=False,
        font=dict(size=16, color="rgb(214, 39, 40)"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgb(214, 39, 40)",
        borderwidth=2,
        borderpad=10
    )
    
    # Top 5 mejores
    top5 = df.nlargest(5, 'quality_score')[['archivo', 'quality_score']]
    top5_text = "<b>Top 5 Mejores Transcripciones:</b><br>" + "<br>".join(
        [f"{i+1}. {row['archivo'][:30]}... ({row['quality_score']:.1f})" 
         for i, row in top5.iterrows()]
    )
    
    fig.add_annotation(
        text=top5_text,
        xref="paper", yref="paper",
        x=0.25, y=0.3, showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(200, 255, 200, 0.8)",
        bordercolor="green",
        borderwidth=1,
        borderpad=10,
        align="left"
    )
    
    # Bottom 5 (requieren revision)
    bottom5 = df.nsmallest(5, 'quality_score')[['archivo', 'quality_score']]
    bottom5_text = "<b>Top 5 Menor Confianza (Revisar):</b><br>" + "<br>".join(
        [f"{i+1}. {row['archivo'][:30]}... ({row['quality_score']:.1f})" 
         for i, row in bottom5.iterrows()]
    )
    
    fig.add_annotation(
        text=bottom5_text,
        xref="paper", yref="paper",
        x=0.75, y=0.3, showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 200, 200, 0.8)",
        bordercolor="red",
        borderwidth=1,
        borderpad=10,
        align="left"
    )
    
    fig.update_layout(
        title="Dashboard de Metricas de Transcripcion",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=600,
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    
    return fig


def generate_all_visualizations(transcription_dir: str = None):
    """
    Genera todas las visualizaciones de metricas de transcripcion
    
    Args:
        transcription_dir: Directorio con archivos de transcripcion
    
    Returns:
        Diccionario con todas las figuras
    """
    # Cargar datos
    df = load_transcription_metrics(transcription_dir)
    
    if df.empty:
        print("No hay datos para visualizar")
        return {}
    
    print("\nGenerando visualizaciones interactivas...")
    
    figures = {
        'dashboard': generate_summary_dashboard(df),
        'quality_distribution': plot_quality_distribution(df),
        'confidence_scores': plot_confidence_scores(df),
        'duration_vs_quality': plot_duration_vs_quality(df),
        'processing_efficiency': plot_processing_efficiency(df),
        'preprocessing_effectiveness': plot_preprocessing_effectiveness(df),
        'speaker_analysis': plot_speaker_analysis(df)
    }
    
    print(f"{len(figures)} visualizaciones generadas")
    
    return figures, df


if __name__ == "__main__":
    # Ejemplo de uso
    figures, df = generate_all_visualizations()
    
    # Mostrar figuras
    for name, fig in figures.items():
        print(f"\nMostrando: {name}")
        fig.show()
