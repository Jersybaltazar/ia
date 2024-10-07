from typing import Dict, Any, Optional , List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from base.base_agent import BaseAgent, AgentResponse
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class PrecipitationData(BaseModel):
    """Modelo de datos para precipitación general"""
    average_precipitation_mm: float = Field(ge=0)
    total_precipitation_mm: float = Field(ge=0)
    intensity_distribution: Dict[str, int]  # Conteo de intensidades: light, moderate, heavy
    analysis_date: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "average_precipitation_mm": 15.2,
                "total_precipitation_mm": 1520.0,
                "intensity_distribution": {"light": 50, "moderate": 30, "heavy": 20},
                "analysis_date": "2023-10-05T12:00:00Z"
            }
        }
class PrecipitationAgent(BaseAgent):
    """
    Agente especializado en análisis de precipitación y pronóstico
    
    Este agente:
    1. Lee datos de precipitación ya preparados desde un archivo CSV.
    2. Analiza patrones de precipitación.
    3. Proporciona los resultados técnicos del análisis..
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3):
        super().__init__(model_name, temperature)
        self.data_file_path = 'precipitation.csv'

    def _get_capabilities(self) -> List[str]:
        """Define las capacidades específicas del agente"""
        return ["precipitation analysis"]
        
    async def validate_input(self, context: Dict[str, Any]) -> bool:    
        return True
    

    async def _load_prepared_data(self) -> pd.DataFrame:
        """Carga los datos de precipitación preparados desde un archivo CSV"""
        try:
            data = pd.read_csv(self.data_file_path)
            logger.info("Datos de precipitación cargados exitosamente")
            return data
        except FileNotFoundError:
            logger.error(f"Archivo de datos no encontrado: {self.data_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error al cargar los datos: {str(e)}")
            raise

    
    async def _analyze_precipitation(self, data: pd.DataFrame) -> PrecipitationData:
        """Analiza todos los datos de precipitación y genera los resultados técnicos generales"""
        # Verificar si existe la columna 'precipitation' en mm
        if 'precipitation' not in data.columns:
            logger.error("La columna 'precipitation' no se encuentra en los datos")
            raise ValueError("La columna 'precipitation' es necesaria para el análisis")

        # Convertir la columna 'precipitation' a tipo numérico
        data['precipitation'] = pd.to_numeric(data['precipitation'], errors='coerce')

        # Eliminar filas con valores faltantes o inválidos
        data = data.dropna(subset=['precipitation'])

        # Calcular estadísticas generales
        average_precipitation_mm = data['precipitation'].mean()
        total_precipitation_mm = data['precipitation'].sum()

        # Calcular la distribución de intensidades
        data['intensity'] = data['precipitation'].apply(self._calculate_intensity)
        intensity_distribution = data['intensity'].value_counts().to_dict()

        return PrecipitationData(
            average_precipitation_mm=average_precipitation_mm,
            total_precipitation_mm=total_precipitation_mm,
            intensity_distribution=intensity_distribution
        )

    def _calculate_intensity(self, precipitation: float) -> str:
        """Calcula la intensidad de la precipitación"""
        if precipitation < 0.5:
            return "light"
        elif precipitation < 4.0:
            return "moderate"
        else:
            return "heavy"        
    
    async def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Procesa la solicitud de análisis de precipitación.

        Args:
            context: Diccionario con la información necesaria:
                - latitude: float
                - longitude: float
                - date: str (YYYY-MM-DD)
                - forecast_days: int

        Returns:
            AgentResponse con los datos de precipitación y recomendaciones.
        """
        try:
            #if not await self.validate_input(context):
                #return self._format_error("Contexto inválido: faltan campos requeridos o son incorrectos")

            # Cargar datos preparados
            data = await self._load_prepared_data()

            # Analizar datos
            precipitation_data = await self._analyze_precipitation(data)

            # Generar recomendaciones
            #recommendations = await self._generate_recommendations(precipitation_data)

            return AgentResponse(
                success=True,
                data={
                    "precipitation_data": precipitation_data.model_dump(),
                }
            )

        except Exception as e:
            logger.error(f"Error en el procesamiento del agente de precipitación: {str(e)}")
            return self._format_error(f"Error en el procesamiento: {str(e)}")