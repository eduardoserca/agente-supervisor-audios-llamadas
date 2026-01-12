"""
Utilidad para cargar y parsear archivos JSON de prompts/criterios
"""
import json
from pathlib import Path
from typing import Dict


class PromptLoader:
    """Cargador de archivos JSON de criterios"""
    
    def __init__(self, json_path: str):
        """
        Inicializa el cargador
        
        Args:
            json_path: Ruta al archivo JSON
        """
        self.json_path = Path(json_path)
        self.criteria = None
        self.load()
    
    def load(self) -> Dict:
        """
        Carga el archivo JSON
        
        Returns:
            Diccionario con criterios
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.criteria = json.load(f)
        
        return self.criteria
    
    def get_rule(self, rule_name: str) -> str:
        """
        Obtiene una regla específica
        
        Args:
            rule_name: Nombre de la regla (ej: "R2_empatia_claridad")
        
        Returns:
            Texto de la regla
        """
        return self.criteria.get("reglas", {}).get(rule_name, "")
    
    def get_all_rules(self) -> Dict[str, str]:
        """
        Obtiene todas las reglas
        
        Returns:
            Diccionario con todas las reglas
        """
        return self.criteria.get("reglas", {})
    
    def get_weights(self) -> Dict[str, float]:
        """
        Obtiene las ponderaciones de las reglas
        
        Returns:
            Diccionario con ponderaciones
        """
        return self.criteria.get("ponderaciones", {})
    
    def get_definitions(self) -> Dict:
        """
        Obtiene las definiciones
        
        Returns:
            Diccionario con definiciones
        """
        return self.criteria.get("definiciones", {})
    
    def get_version(self) -> str:
        """
        Obtiene la versión del criterio
        
        Returns:
            String con versión
        """
        return self.criteria.get("version", "unknown")
    
    def get_criterion_name(self) -> str:
        """
        Obtiene el nombre del criterio
        
        Returns:
            String con nombre del criterio
        """
        return self.criteria.get("criterio", "")
