from typing import Dict, Any
from pydantic import BaseModel
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from base.base_agent import AgentResponse
logger = logging.getLogger(__name__)

class ResponseSynthesizer:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self._initialize_prompts()

    def _initialize_prompts(self):
        """Inicializa los prompts para diferentes tipos de síntesis"""
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un experto agrícola que ayuda a interpretar datos técnicos 
            y convertirlos en recomendaciones prácticas para agricultores. 
            Tu objetivo es proporcionar respuestas claras y accionables."""),
            ("user", """Consulta del agricultor: {query}

            Resultados técnicos:
            {technical_results}

            Por favor, proporciona una respuesta clara y práctica que el agricultor pueda entender y aplicar.""")
        ])

    async def create_response(
        self, 
        query: str, 
        agent_responses: Dict[str, AgentResponse]
    ) -> str:
        """Crea una respuesta coherente y comprensible para el agricultor"""
        try:
            # Filtrar y organizar resultados técnicos
            technical_results = self._format_technical_results(agent_responses)
            
            # Generar respuesta usando el LLM
            messages = self.synthesis_prompt.format_messages(
                query=query,
                technical_results=technical_results
            )
            
            response = await self.llm.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Error en síntesis de respuesta: {str(e)}", exc_info=True)
            return "Lo siento, hubo un error al procesar los resultados. Por favor, inténtalo de nuevo."

    def _format_technical_results(self, agent_responses: Dict[str, AgentResponse]) -> str:
        """Formatea los resultados técnicos para el prompt"""
        formatted_results = []
        
        for agent_name, response in agent_responses.items():
            if response.success:
                formatted_results.append(f"[{agent_name}]\n{self._format_data(response.data)}")
            else:
                formatted_results.append(f"[{agent_name}] Error: {response.error_message}")
        
        return "\n\n".join(formatted_results)

    def _format_data(self, data: Dict[str, Any]) -> str:
        """Formatea los datos técnicos de manera legible"""
        return "\n".join(f"- {key}: {value}" for key, value in data.items())