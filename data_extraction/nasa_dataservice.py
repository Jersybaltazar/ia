# data_extraction/nasa_data_service.py

from typing import Dict, Any
import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)

class NasaDataService:
    """Servicio para extraer y preparar datos de la NASA."""

    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    async def get_soil_moisture_data(self, latitude: float, longitude: float, date: str) -> Dict[str, Any]:
        """Obtiene datos de humedad del suelo."""
        params = {
            'parameters': 'SOILM',
            'start': date.replace('-', ''),
            'end': date.replace('-', ''),
            'latitude': latitude,
            'longitude': longitude,
            'format': 'JSON'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._prepare_soil_moisture_data(data)
                else:
                    logger.error(f"Error al obtener datos de la NASA: {response.status}")
                    raise Exception(f"Error al obtener datos: {response.status}")

    def _prepare_soil_moisture_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara los datos de humedad del suelo."""
        # Implementar lógica de preparación
        # Por ejemplo, extraer valores necesarios y estructurarlos
        prepared_data = {
            'date': list(raw_data['properties']['parameter']['SOILM'].keys())[0],
            'soil_moisture': list(raw_data['properties']['parameter']['SOILM'].values())[0]
        }
        return prepared_data

    async def get_precipitation_data(self, latitude: float, longitude: float, start_date: str, end_date: str) -> Dict[str, Any]:
        """Obtiene datos de precipitación."""
        params = {
            'parameters': 'PRECTOT',
            'start': start_date,
            'end': end_date,
            'latitude': latitude,
            'longitude': longitude,
            'format': 'JSON'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._prepare_precipitation_data(data)
                else:
                    logger.error(f"Error al obtener datos de la NASA: {response.status}")
                    raise Exception(f"Error al obtener datos: {response.status}")

    def _prepare_precipitation_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara los datos de precipitación."""
        # Implementar lógica de preparación
        prepared_data = {
            'dates': list(raw_data['properties']['parameter']['PRECTOT'].keys()),
            'precipitation': list(raw_data['properties']['parameter']['PRECTOT'].values())
        }
        return prepared_data
