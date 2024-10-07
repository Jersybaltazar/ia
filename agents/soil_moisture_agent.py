from typing import Dict, Any , List , Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from base.base_agent import BaseAgent, AgentResponse
from data_extraction.nasa_dataservice import NasaDataService
from schemas import SoilMoistureData
import logging
import json
import pandas as pd

logger = logging.getLogger(__name__)
"""
class SoilMoistureData(BaseModel):
    Modelo de datos para precipitación
    date: datetime
    precipitation_mm: float = Field(ge=0)
    probability: float = Field(ge=0, le=1)
    intensity: str = Field(default="normal")
    forecast_days: Optional[int] = Field(default=1, ge=1, le=7)

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-03-07T00:00:00Z",
                "precipitation_mm": 25.4,
                "probability": 0.85,
                "intensity": "moderate",
                "forecast_days": 3
            }
        }
"""

class SoilMoistureAgent(BaseAgent):
    """Agente especializado en análisis de humedad del suelo"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3):
        super().__init__(model_name, temperature)
        self.data_file_path = 'export.csv'
        
    def _get_capabilities(self) -> List[str]:
        """Define las capacidades específicas del agente"""
        return ["soil moisture analysis", "irrigation recommendations"]
    
    async def validate_input(self, context: Dict[str, Any]) -> bool:
        """Valida que el contexto tenga la información necesaria"""
        required_fields = ['latitude', 'longitude', 'date']
        return all(field in context for field in required_fields)
    
    async def load_and_process_data(self) -> pd.DataFrame:
        """Carga y procesa los datos desde el archivo CSV"""
        # Cargar el archivo CSV
        data = pd.read_csv(self.data_file_path)
        required_columns = ['nir', 'swir16']
        if not all(column in data.columns for column in required_columns):
            raise ValueError(f"Las columnas {required_columns} son necesarias en los datos.")   
        # Calcular el índice NDWI (Normalized Difference Water Index)
        data['ndwi'] = (data['nir'] - data['swir16']) / (data['nir'] + data['swir16'])
        
        return data
    
    async def analyze_soil_moisture(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analiza los datos de humedad del suelo y prepara los resultados técnicos"""
        # Crear un análisis técnico basado en el NDWI y preparar los resultados
        analysis_results = []

        for _, row in data.iterrows():
            ndwi = row['ndwi']
            status = "high moisture" if ndwi > 0.3 else "low moisture" if ndwi < 0.1 else "moderate moisture"

            result = {
                "lat": row['lat'],
                "lon": row['lon'],
                "ndwi": ndwi,
                "status": status,
                "date": row['scene_date']
            }
            analysis_results.append(result)
        
        return analysis_results
    
    async def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Procesa la solicitud de análisis de humedad del suelo y devuelve los resultados técnicos."""
        try:
            # Cargar y procesar los datos
            data = await self.load_and_process_data()
            
            # Analizar los datos de humedad del suelo
            analysis_results = await self.analyze_soil_moisture(data)
            
            # Formatear los resultados para ser devueltos al coordinador
            return AgentResponse(
                success=True,
                data={"analysis_results": analysis_results},
                error_message=""
            )
        except Exception as e:
            logger.error(f"Error en el procesamiento del agente de humedad: {str(e)}")
            return self._format_error(f"Error en el procesamiento: {str(e)}")

    

