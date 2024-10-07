import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class AgentMetadata(BaseModel):
    """Metadata común para todos los agentes"""
    agent_id: str
    agent_type: str
    version: str = "1.0"
    last_updated: datetime = Field(default_factory=datetime.now)
    capabilities: List[str]

class AgentResponse(BaseModel):
    """Modelo base para las respuestas de los agentes"""
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Método para obtener valores del diccionario data de manera segura
        
        Args:
            key: Clave a buscar
            default: Valor por defecto si la clave no existe
            
        Returns:
            El valor asociado a la clave o el valor por defecto
        """
        return self.data.get(key, default)
    
class BaseAgent(ABC):
    """
    Clase base abstracta para todos los agentes del sistema.
    
    Proporciona:
    1. Gestión de estado y memoria
    2. Interacción con LLM
    3. Logging y monitoreo
    4. Manejo de errores estándar
    5. Validación de datos
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        max_retries: int = 3
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        self.workflow = StateGraph(state_schema=MessagesState)
        self.memory = MemorySaver()
        # Definir cómo se maneja la interacción del modelo
        def call_model(state: MessagesState):
            response = self.llm.invoke(state["messages"])
            # Retornar una lista para que se agregue al historial
            return {"messages": response}
        
         # Añadir el nodo al grafo
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)

        # Compilar la aplicación con la memoria integrada
        self.app = self.workflow.compile(checkpointer=self.memory)

        # Generar un identificador único para el hilo de conversación
        self.thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.metadata = self._initialize_metadata()
        
    def _initialize_metadata(self) -> AgentMetadata:
        """Inicializa los metadatos del agente"""
        return AgentMetadata(
            agent_id=self.__class__.__name__.lower(),
            agent_type=self.__class__.__name__,
            capabilities=self._get_capabilities()
        )
    
    async def interact_with_model(self, input_message: HumanMessage):
        """
        Interactúa con el modelo usando el flujo de trabajo definido.
        Args:
            input_message: Mensaje de entrada (por ejemplo, la consulta del usuario)

        Returns:
            Respuesta generada por el modelo
        """
        # Usar el flujo para enviar el mensaje y obtener la respuesta
        for event in self.app.stream({"messages": [input_message]}, self.config, stream_mode="values"):
            return event["messages"][-1].content
    @abstractmethod

    def _get_capabilities(self) -> List[str]:
        """Define las capacidades específicas del agente"""
        pass
    
    @abstractmethod
    async def validate_input(self, context: Dict[str, Any]) -> bool:
        """Valida el contexto de entrada"""
        pass
    
    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> AgentResponse:
        """Procesa la solicitud principal"""
        pass
    
    async def _llm_interaction(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 500
    ) -> str:
        """
        Maneja la interacción con el LLM de manera estandarizada
        
        Args:
            prompt: Prompt principal
            system_message: Mensaje de sistema opcional
            max_tokens: Límite de tokens
            
        Returns:
            str: Respuesta del LLM
        """
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text.strip()
        except Exception as e:
            logger.error(f"Error en interacción con LLM: {str(e)}")
            raise
    
    def _format_error(self, message: str, details: Optional[Dict] = None) -> AgentResponse:
        """
        Formatea errores de manera consistente
        
        Args:
            message: Mensaje de error principal
            details: Detalles adicionales del error
            
        Returns:
            AgentResponse con información del error
        """
        error_data = {
            "error_type": self.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        logger.error(f"Error en {self.metadata.agent_type}: {message}", 
                    extra={"error_data": error_data})
        
        return AgentResponse(
            success=False,
            data={},
            error_message=message,
            metadata=error_data
        )
    
    async def update_memory(self, context: Dict[str, Any], response: AgentResponse):
        """
        Actualiza la memoria del agente con el contexto y respuesta actual
        
        Args:
            context: Contexto de la interacción
            response: Respuesta generada
        """
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "response": response.model_dump(),
            "agent": self.metadata.model_dump()
        }
        
        self.memory.chat_memory.add_user_message(
            json.dumps(memory_entry, default=str)
        )
    
    async def get_memory_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de la memoria del agente
        
        Returns:
            Dict con el resumen de la memoria
        """
        messages = self.memory.chat_memory.messages
        return {
            "message_count": len(messages),
            "last_interaction": messages[-1].content if messages else None,
            "agent_metadata": self.metadata.dict()
        }