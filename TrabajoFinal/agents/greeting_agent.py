"""
Agente de calidad y cumplimiento  con LangChain
Analiza si el asesor cumple con las reglas definidas en el JSON de criterios
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import config
from agents.prompt_loader import PromptLoader

# Importar LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage, SystemMessage
import re    


# Modelos Pydantic para structured output
class RuleAnalysis(BaseModel):
    """Análisis de una regla individual"""
    cumple: bool = Field(description="Si la regla se cumple o no")
    razon: str = Field(description="Explicación breve y concisa del resultado")
    evidencia: str = Field(description="Cita textual o referencia de la transcripción")
    score: int = Field(ge=0, le=100, description="Puntuación de 0 a 100")


class QualityAnalysisOutput(BaseModel):
    """Estructura completa del análisis de calidad"""
    rule_analysis: Dict[str, RuleAnalysis] = Field(
        description="Análisis de cada regla, con el ID de la regla como clave"
    )


class QualityAgent:
    """Agente para analizar calidad y cumplimiento de criterios usando LangChain"""
    
    def __init__(self, 
                 criteria_path: str = None,
                 use_ai: bool = True,
                 model_name: str = None,
                 base_url: str = None):
        """
        Inicializa el agente de calidad
        
        Args:
            criteria_path: Ruta al archivo JSON de criterios
            use_ai: Usar LLM para análisis inteligente
            model_name: Nombre del modelo a usar
            base_url: URL base de la API del LLM
        """
        # Cargar criterios
        criteria_path = criteria_path or str(config.CRITERIOS_JSON)
        self.prompt_loader = PromptLoader(criteria_path)
        
        # Cargar prompts
        self.prompts = self._load_prompts()
        
        # Configurar LLM
        self.use_ai = use_ai and config.OPENAI_API_KEY
        self.model_name = model_name or config.GPT_MODEL
        self.base_url = base_url or config.OPENAI_BASE_URL
        
        # Usar LangChain
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            openai_api_key=config.OPENAI_API_KEY,
            openai_api_base=self.base_url
        )
        print(f"Agente configurado con LangChain")
        print(f"   - Modelo: {self.model_name}")
        print(f"   - Base URL: {self.base_url}")
        self._test_llm_connection()
        
            
    def _test_llm_connection(self):
        """Prueba la conexion con el LLM"""
        try:
            print("   Probando conexion con LLM...", end=" ")
            self.llm.invoke("Test")
            print("Conexion exitosa (LangChain)")
        except Exception as e:
            print(f"Error: {str(e)[:50]}")
            print("   El analisis continuara pero puede fallar")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Carga los prompts desde el archivo de configuración"""
        try:
            with open(config.QUALITY_PROMPTS_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando prompts: {e}. Usando defaults.")
            return {
                "system_prompt": "Eres un experto auditor de calidad en servicio al cliente. Sé estricto y objetivo. \nResponde UNICAMENTE con un objeto JSON que siga la estructura solicitada, sin explicaciones adicionales fuera del JSON.",
                "user_prompt_template": "Analiza detenidamente la siguiente transcripción de una llamada de servicio al cliente y evalúa cada una de las reglas proporcionadas.\n\nTranscripción:\n{transcription_text}\n\nREGLAS A EVALUAR:\n{rules_text}"
            }
    
    def analyze_all_rules_batch_OLD(self, transcription_text: str, rules: Dict[str, str]) -> Dict[str, Dict]:
        """
        Analiza todas las reglas usando LangChain con structured output
        
        Args:
            transcription_text: Texto completo de la transcripción
            rules: Diccionario con todas las reglas {rule_id: rule_description}
        
        Returns:
            Diccionario con análisis de cumplimiento por regla
        """
        if not self.llm:
            return {rule_id: {"cumple": False, "razon": "IA no disponible", "score": 0, "evidencia": ""} 
                    for rule_id in rules.keys()}
        
        # Construir lista de reglas para el prompt
        rules_text = "\n\n".join([f"**{rule_id}**:\n{desc}" for rule_id, desc in rules.items()])
        
        # Crear una versión minificada de las instrucciones de formato para ahorrar tokens
        compact_format_instructions = (
            "Responde exclusivamente en formato JSON con esta estructura:\n"
            "{\"rule_analysis\": {\"ID_REGLA\": {\"cumple\": bool, \"razon\": \"string\", \"evidencia\": \"string\", \"score\": int}}}"
        )
        
        # Crear prompt template (sin el parser verbose de LangChain)
        prompt_template = PromptTemplate(
            template=self.prompts["user_prompt_template"] + "\n\n{format_instructions}",
            input_variables=["transcription_text", "rules_text"],
            partial_variables={"format_instructions": compact_format_instructions}
        )
        
        try:
            print(f"   Llamando al LLM con LangChain ({self.model_name})...", end=" ")
            
            # Formatear prompt (eliminamos la truncación agresiva para dejar que el servidor maneje su límite)
            formatted_prompt = prompt_template.format(
                transcription_text=transcription_text,
                rules_text=rules_text
            )
            
            # Llamar al LLM
            #print(f"Formateando prompt...{formatted_prompt}")
            messages = [
                SystemMessage(content=self.prompts["system_prompt"]),
                HumanMessage(content=formatted_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            print(f"Respuesta recibida")
            
            # Parsear con Pydantic de manera robusta
            analysis = {}
            try:
                # Extraer JSON de la respuesta de manera flexible
                content = response.content
                if not content or not content.strip():
                    raise ValueError("El LLM devolvió una respuesta vacía.")
                
                # Intentar encontrar bloques de código JSON
                if "```json" in content:
                    content = content.split("```json")[-1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[-1].split("```")[0].strip()
                
                # Si no hay bloques de código, intentar buscar las llaves { }
                if not content.startswith("{") and "{" in content:
                    content = "{" + content.split("{", 1)[1]
                if not content.endswith("}") and "}" in content:
                    content = content.rsplit("}", 1)[0] + "}"

                # Cargar JSON crudo para inspección
                try:
                    raw_json = json.loads(content)
                except json.JSONDecodeError as jde:
                    print(f"\n   [!] Error decodificando JSON. Respuesta cruda del LLM:\n{'-'*40}\n{response.content}\n{'-'*40}")
                    raise jde
                
                # Si el LLM devolvió los resultados directamente sin el wrapper "rule_analysis"
                if isinstance(raw_json, dict) and "rule_analysis" not in raw_json:
                    if any(rule_id in raw_json for rule_id in rules.keys()):
                        raw_json = {"rule_analysis": raw_json}
                
                # Validar con Pydantic si es posible
                try:
                    parsed_output = QualityAnalysisOutput.parse_obj(raw_json)
                    analysis = {rule_id: analysis_item.dict() for rule_id, analysis_item in parsed_output.rule_analysis.items()}
                except Exception as pydantic_err:
                    print(f"\n   Error en validacion Pydantic: {pydantic_err}. Usando JSON crudo.")
                    if isinstance(raw_json, dict) and "rule_analysis" in raw_json:
                        analysis = raw_json["rule_analysis"]
                    else:
                        analysis = raw_json
                
            except Exception as parse_error:
                print(f"\n   Error crítico parseando respuesta del LLM: {parse_error}")
                # analysis queda vacío y se llenará en el loop de seguridad de abajo
            
            # Asegurar que todas las reglas estén presentes
            for rule_id in rules.keys():
                if rule_id not in analysis:
                    analysis[rule_id] = {
                        "cumple": False,
                        "razon": "No evaluada por la IA",
                        "score": 0,
                        "evidencia": ""
                    }
            
            return analysis
        
        except Exception as e:
            error_msg = str(e)
            print(f"\n   Error analizando con LangChain: {error_msg}")
            
            if "context length" in error_msg.lower() or "token" in error_msg.lower():
                print("\n   [!] ERROR DE CONTEXTO DETECTADO:")
                print("   La transcripción es demasiado larga para la configuración actual de tu LLM.")
                print("   SOLUCIÓN: En LM Studio, sube el 'Context Length' (ej: a 16384) y recarga el modelo.")
            else:
                print(f"   Verifica que tu servidor LLM este ejecutandose correctamente.")
            
            # Fallback: retornar estructura básica
            return {rule_id: {"cumple": False, "razon": f"Error de análisis: {error_msg[:100]}", "score": 0, "evidencia": ""} 
                    for rule_id in rules.keys()}

    def analyze_all_rules_batch(self, transcription_text: str, rules: Dict[str, str]) -> Dict[str, Dict]:
        """
        Versión ultra-robusta: Extrae el JSON, sanitiza tildes en llaves y corrige comillas internas.
        """

        if not self.use_ai or not self.llm:
            return {rule_id: {"cumple": False, "razon": "IA no disponible", "score": 0, "evidencia": ""} 
                    for rule_id in rules.keys()}
        
        rules_text = "\n\n".join([f"**{rule_id}**:\n{desc}" for rule_id, desc in rules.items()])
        
        prompt_template = PromptTemplate(
            template=self.prompts["user_prompt_template"],
            input_variables=["transcription_text", "rules_text"],
        )
        
        try:
            print(f"   Llamando al LLM ({self.model_name})...", end=" ")
            formatted_prompt = prompt_template.format(transcription_text=transcription_text, rules_text=rules_text)
            
            response = self.llm.invoke([
                SystemMessage(content=self.prompts["system_prompt"]),
                HumanMessage(content=formatted_prompt)
            ])

            print(response)
            
            raw_content = response.content
            print(f"Respuesta recibida")
            
            analysis = {}
            try:
                # --- PASO 1: EXTRAER EL BLOQUE JSON (Búsqueda codiciosa) ---
                # Buscamos desde el primer '{' hasta el ÚLTIMO '}'
                match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
                if not match:
                    raise ValueError("No se encontró JSON en la respuesta")
                
                content = match.group(0)

                # --- PASO 2: LIMPIEZA DE "RAZÓN" (Tildes en llaves) ---
                content = content.replace('"razón":', '"razon":')
                
                # --- PASO 3: LIMPIEZA DE NOTAS ADYACENTES ---
                # Si Llama pegó "Nota:" justo después de la llave, lo cortamos
                if "}" in content:
                    last_brace_index = content.rfind("}")
                    content = content[:last_brace_index + 1]

                # --- PASO 4: CORRECCIÓN DE COMILLAS DOBLES INTERNAS ---
                # Este regex busca comillas dobles que NO están seguidas por : , } o ]
                # Ayuda a evitar el error 'Expecting , delimiter'
                def clean_internal_quotes(m):
                    return m.group(0).replace('"', "'")
                
                # Buscamos el contenido de los strings y cambiamos comillas dobles por simples
                content = re.sub(r':\s*"(.*?)"(?=[,}])', lambda m: m.group(0).replace('"', '\"'), content)

                # --- PASO 5: CARGAR JSON ---
                raw_json = json.loads(content)
                
                # Normalizar si falta el wrapper
                if "rule_analysis" not in raw_json:
                    if any(k.startswith('R') for k in raw_json.keys()):
                        raw_json = {"rule_analysis": raw_json}
                
                # --- PASO 6: VALIDAR CON PYDANTIC ---
                try:
                    parsed_output = QualityAnalysisOutput.parse_obj(raw_json)
                    analysis = {rid: item.dict() for rid, item in parsed_output.rule_analysis.items()}
                except Exception as p_err:
                    print(f"\n   [!] Error validando esquema: {p_err}. Usando datos crudos.")
                    analysis = raw_json.get("rule_analysis", raw_json)
                
            except Exception as parse_error:
                print(f"\n   [!] Error crítico decodificando: {parse_error}")
                print(f"   Fragmento del error: {raw_content[-100:]}")
                analysis = {}
            
            # Rellenar reglas faltantes para no romper el global_score
            for rule_id in rules.keys():
                if rule_id not in analysis:
                    analysis[rule_id] = {"cumple": False, "razon": "Error formato IA", "score": 0, "evidencia": ""}
            
            return analysis
            
        except Exception as e:
            print(f"\n   Error general: {str(e)}")
            return {rid: {"cumple": False, "razon": "Error sistema", "score": 0, "evidencia": ""} for rid in rules.keys()}




    def analyze_transcription(self, transcription_data: Dict) -> Dict:
        """
        Analiza una transcripción completa evaluando todas las reglas
        
        Args:
            transcription_data: Datos de transcripción (del TranscriptionAgent)
        
        Returns:
            Diccionario con análisis completo
        """
        full_text = transcription_data.get("full_text", "")
        if not full_text:
            return {rule_id: {"cumple": False, "razon": "No hay texto", "score": 0, "evidencia": ""} 
                    for rule_id in self.prompt_loader.get_all_rules().keys()}
        
        all_rules = self.prompt_loader.get_all_rules()
        weights = self.prompt_loader.get_weights()
        
        # Filtrar solo las reglas evaluables (que empiezan con R y tienen peso)
        evaluable_rules = {
            rule_id: rule_desc 
            for rule_id, rule_desc in all_rules.items() 
            if rule_id.startswith("R") and not rule_id.endswith("_nota_caso_interrupcion_cliente")
        }
        
        print(f"\nEvaluando {len(evaluable_rules)} reglas en una sola llamada al LLM...")
        
        # Evaluar todas las reglas en batch (una sola llamada al LLM)
        rule_results = self.analyze_all_rules_batch(full_text, evaluable_rules)
        print(f"   Analisis completado")
        
        # Calcular score ponderado
        total_score = 0
        total_weight = 0
        
        for rule_id, analysis in rule_results.items():
            weight = weights.get(rule_id, 0)
            score = analysis.get("score", 0)
            
            total_score += (score * weight)
            total_weight += weight

        # Normalizar score final a 0-100
        final_score = total_score if total_weight > 0 else 0
        
        # Construir resultado
        result = {
            "audio_file": transcription_data.get("audio_file", "unknown"),
            "model_used": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "criteria_version": self.prompt_loader.get_version(),
            "rule_analysis": rule_results,
            "global_score": round(final_score, 2),
            "compliance": {
                "global_result": "CUMPLE" if final_score >= 80 else "NO CUMPLE",
                "score": round(final_score, 2)
            }
        }
        
        return result
    
    def save_analysis(self, result: Dict, output_path: str):
        """
        Guarda el análisis en archivo JSON
        
        Args:
            result: Diccionario con resultado de análisis
            output_path: Ruta del archivo de salida
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Análisis guardado en: {output_path}")
    
    def process_transcription_file(self, transcription_path: str, 
                                   output_path: str = None) -> Dict:
        """
        Procesa un archivo de transcripción
        
        Args:
            transcription_path: Ruta al archivo JSON de transcripción
            output_path: Ruta de salida (opcional)
        
        Returns:
            Diccionario con análisis
        """
        # Cargar transcripción
        with open(transcription_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        # Analizar
        result = self.analyze_transcription(transcription_data)
        
        # Guardar si se especifica ruta
        if output_path:
            self.save_analysis(result, output_path)
        else:
            # Generar ruta por defecto incluyendo el modelo para evitar colisiones
            trans_path = Path(transcription_path)
            safe_model = self.model_name.split('/')[-1].replace(':', '_')
            default_output = config.GREETING_DIR / f"{trans_path.stem}_{safe_model}_quality_analysis.json"
            self.save_analysis(result, str(default_output))
        
        return result
    
    def process_directory(self, transcription_dir: str = None,
                         output_dir: str = None) -> List[Dict]:
        """
        Procesa todos los archivos de transcripción en un directorio
        
        Args:
            transcription_dir: Directorio con transcripciones
            output_dir: Directorio de salida
        """
        transcription_dir = Path(transcription_dir or config.TRANSCRIPTION_DIR)
        output_dir = Path(output_dir or config.GREETING_DIR)
        
        # Buscar archivos de transcripción
        transcription_files = list(transcription_dir.glob("*_transcription.json"))
        
        if not transcription_files:
            print(f"No se encontraron archivos de transcripción en {transcription_dir}")
            return []
        
        print(f"Encontrados {len(transcription_files)} archivos de transcripción")
        
        results = []
        for i, trans_file in enumerate(transcription_files, 1):
            print(f"\n{'='*60}")
            print(f"Analizando {i}/{len(transcription_files)}: {trans_file.name}")
            print(f"{'='*60}")
            
            try:
                # Procesar transcripción
                # El output_path ahora se genera automáticamente en process_transcription_file si es None
                result = self.process_transcription_file(str(trans_file))
                
                results.append(result)
                
                print(f"Completado: {trans_file.name}")
                print(f"  - Modelo: {self.model_name}")
                print(f"  - Score Global: {result['global_score']}/100")
                
            except Exception as e:
                print(f"Error analizando {trans_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"Analisis completado: {len(results)}/{len(transcription_files)} exitosos")
        print(f"{'='*60}")
        
        return results

# Alias para compatibilidad hacia atrás
GreetingAgent = QualityAgent

if __name__ == "__main__":
    # Ejemplo de uso
    agent = QualityAgent()
    results = agent.process_directory()
