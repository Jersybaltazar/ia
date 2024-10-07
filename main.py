import asyncio
import logging
from dotenv import load_dotenv
from typing import Dict, Any
from core.coordinator import AgentCoordinator
from core.synthesizer import ResponseSynthesizer
import logging


logger = logging.getLogger(__name__)

async def process_query(query: str, context: Dict[str, Any]) -> str:
    """
    Procesa una consulta del agricultor
    Args:
        query: Pregunta o consulta del agricultor
        context: Información contextual (ubicación, fecha, etc.)
    Returns:
        str: Respuesta final para el agricultor
    """
    try:
        # Inicializar componentes
        coordinator = AgentCoordinator()
        synthesizer = ResponseSynthesizer()
        
        # Obtener plan de acción
        logger.info(f"Procesando consulta: {query}")
        plan = await coordinator.create_action_plan(query, context)
        
        # Ejecutar agentes necesarios
        responses = await coordinator.execute_agents(plan, context)
        
        # Sintetizar respuesta final
        final_response = await synthesizer.create_response(query, responses)
        
        logger.info("Consulta procesada exitosamente")
        return final_response
        
    except Exception as e:
        logger.error(f"Error procesando la consulta: {str(e)}")
        return "Lo siento, ha ocurrido un error procesando tu consulta. Por favor, inténtalo de nuevo más tarde."
    
async def main():
    # Ejemplo de uso
    query = input("¿Cuál es tu consulta sobre el riego? ")
    context = {
        "latitude": 19.4326,
        "longitude": -99.1332,
        "date": "2024-03-07"
    }
    
    response = await process_query(query, context)
    print("\nRespuesta:", response)

if __name__ == "__main__":
    asyncio.run(main())