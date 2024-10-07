from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SoilMoistureData(BaseModel):
    """
    Modelo de datos que estructura la información de humedad del suelo.
    """

    date: datetime = Field(
        ...,
        description="Fecha de los datos."
    )
    current_level: float = Field(
        ...,
        gt=0,
        lt=100,
        description="Nivel actual de humedad del suelo (%)."
    )
    optimal_level: float = Field(
        ...,
        gt=0,
        lt=100,
        description="Nivel óptimo de humedad para el cultivo (%)."
    )
    trend: str = Field(
        default="estable",
        description='Tendencia de la humedad (por ejemplo, "ascendente", "descendente", "estable").'
    )
