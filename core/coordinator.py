from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
from base.base_agent import AgentResponse
from agents.soil_moisture_agent import SoilMoistureAgent
from agents.precipitation_agent import PrecipitationAgent
# Importar otros agentes según se vayan creando

logger = logging.getLogger(__name__)

class ExecutionPlan(BaseModel):
    """Modelo para el plan de ejecución"""
    query_type: str
    required_agents: List[str]
    execution_order: List[str]
    priority: str = "normal"
    estimated_completion_time: float
    context_requirements: List[str]

class CoordinatorResponse(BaseModel):
    """Modelo para la respuesta del coordinador"""
    success: bool
    plan: Optional[ExecutionPlan]
    agent_responses: Dict[str, AgentResponse] = Field(default_factory=dict)
    execution_time: float = 0.0
    error_message: str = ""

    def items(self):
        return self.agent_responses.items()
    
class AgentExecutionResult(BaseModel):
    """Modelo para el resultado de ejecución de un agente"""
    agent_name: str
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""
    execution_time: float = 0.0

class AgentCoordinator:
    """
    Coordinador central del sistema multi-agente
    
    Responsabilidades:
    1. Analizar consultas y determinar agentes necesarios
    2. Crear y ejecutar planes de acción
    3. Gestionar la comunicación entre agentes
    4. Manejar errores y recuperación
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.available_agents = {
            "soil_moisture": SoilMoistureAgent,
            "precipitation": PrecipitationAgent
        }
        self.cache = {} 
    
    async def create_action_plan(self, query: str, context: Dict[str, Any]) -> ExecutionPlan:
        """
        Analiza la consulta y crea un plan de acción
        
        Args:
            query: Consulta del usuario
            context: Contexto de la consulta
            
        Returns:
            ExecutionPlan con la estrategia a seguir
        """
        system_prompt = """
        Eres un coordinador experto en agricultura que analiza consultas y determina 
        qué agentes especializados son necesarios. Los agentes disponibles son:
        
        - soil_moisture: Analiza la humedad del suelo
        - precipitation: Analiza precipitaciones y pronósticos
        
        Determina:
        1. Tipo de consulta
        2. Agentes necesarios
        3. Orden de ejecución
        4. Prioridad
        5. Requisitos de contexto
        
        Responde en formato JSON siguiendo el esquema:
        {
            "query_type": "tipo_de_consulta",
            "required_agents": ["agente1", "agente2"],
            "execution_order": ["agente1", "agente2"],
            "priority": "normal|high|low",
            "estimated_completion_time": tiempo_estimado_en_segundos,
            "context_requirements": ["requisito1", "requisito2"]
        }
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Consulta: {query}\nContexto: {context}")
            ]
            
            response = await self.llm.ainvoke(messages)
            print(response.content)
            plan_dict = json.loads(response.content)
            
            return ExecutionPlan(**plan_dict)
            
        except Exception as e:
            logger.error(f"Error creando plan de acción: {str(e)}")
            raise
    
    async def execute_agents(
        self,
        plan: ExecutionPlan,
        context: Dict[str, Any]
    ) -> CoordinatorResponse:
        """
        Ejecuta los agentes según el plan establecido
        
        Args:
            plan: Plan de ejecución
            context: Contexto para los agentes
            
        Returns:
            CoordinatorResponse con resultados de todos los agentes
        """
        start_time = datetime.now()
        responses = {}
        
        try:
            # Validar requisitos de contexto
            missing_requirements = [
                req for req in plan.context_requirements 
                if req not in context
            ]
            if missing_requirements:
                raise ValueError(f"Faltan requisitos en el contexto: {missing_requirements}")
            
            # Ejecutar agentes en el orden especificado
            for agent_name in plan.execution_order:
                if agent_name not in self.available_agents:
                    logger.warning(f"Agente {agent_name} no disponible")
                    continue
                    
                agent_class = self.available_agents[agent_name]
                agent = agent_class()
                
                logger.info(f"Ejecutando agente: {agent_name}")
                response = await agent.process(context)
                responses[agent_name] = response
                
                # Si un agente crítico falla, detener la ejecución
                if not response.success and plan.priority == "high":
                    raise Exception(f"Fallo crítico en agente {agent_name}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return CoordinatorResponse(
                success=True,
                plan=plan,
                agent_responses=responses,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error en ejecución de agentes: {str(e)}")
            return CoordinatorResponse(
                success=False,
                plan=plan,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    async def execute_agents_parallel(
        self,
        plan: ExecutionPlan,
        context: Dict[str, Any]    
    ) -> List[AgentExecutionResult]:
        """
        Ejecuta los agentes en paralelo cuando es posible
        """
        async def execute_single_agent(agent_name: str) -> AgentExecutionResult:
            start_time = datetime.now()
            try:
                # Verificar cache
                cache_key = f"{agent_name}:{context['date']}"
                if cache_key in self.cache:
                    logger.info(f"Usando datos en cache para {agent_name}")
                    return self.cache[cache_key]

                if agent_name not in self.available_agents:
                    raise ValueError(f"Agente {agent_name} no disponible")

                agent_class = self.available_agents[agent_name]
                agent = agent_class()
                
                response = await agent.process(context)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = AgentExecutionResult(
                    agent_name=agent_name,
                    success=response.success,
                    data=response.data,
                    execution_time=execution_time
                )

                # Guardar en cache
                self.cache[cache_key] = result
                return result
            
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                return AgentExecutionResult(
                    agent_name=agent_name,
                    success=False,
                    error_message=str(e),
                    execution_time=execution_time
                )
            # Agrupar agentes por dependencias
        independent_agents = []
        dependent_agents = []

        for agent in plan.required_agents:
            if self._has_dependencies(agent, plan):
                dependent_agents.append(agent)
            else:
                independent_agents.append(agent)

        # Ejecutar agentes independientes en paralelo
        tasks = [execute_single_agent(agent) for agent in independent_agents]
        independent_results = await asyncio.gather(*tasks)

        # Ejecutar agentes dependientes en secuencia
        dependent_results = []
        for agent in dependent_agents:
            result = await execute_single_agent(agent)
            dependent_results.append(result)

        return independent_results + dependent_results
        
    async def _handle_agent_failure(
        self,
        agent_name: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[AgentResponse]:
        """
        Maneja fallos de agentes e intenta recuperarse
        
        Args:
            agent_name: Nombre del agente que falló
            error: Error ocurrido
            context: Contexto de la ejecución
            
        Returns:
            Optional[AgentResponse]: Respuesta recuperada o None
        """
        logger.error(f"Fallo en agente {agent_name}: {str(error)}")
        
        # Intentar estrategias de recuperación
        try:
            agent_class = self.available_agents[agent_name]
            backup_agent = agent_class(temperature=0.5)  # Configuración más flexible
            return await backup_agent.process(context)
        except Exception as backup_error:
            logger.error(f"Fallo en recuperación de {agent_name}: {str(backup_error)}")
            return None