# Aprendizaje: react-web-search.ipynb

## 📋 Resumen

**Notebook:** react-web-search.ipynb (78KB adaptado)
**Tema:** ReAct Agents + Web Search con Tavily
**Estado:** ✅ Adaptado y funcional
**Fecha:** 2025-11-07

### 🎯 Objetivo
Aprender **ReAct (Reason + Act)** - agentes que razonan sobre qué acción tomar y ejecutan búsquedas web para obtener información actualizada.

---

## 🔧 Adaptaciones Realizadas

### 1. LLM (OpenAI → Groq)
```python
# ❌ ANTES
from langchain.chat_models import init_chat_model
llm = init_chat_model("openai:gpt-5-nano")

# ✅ DESPUÉS
from langchain_groq import ChatGroq
llm = ChatGroq(model=os.environ["OPENAI_MODEL"], temperature=0)
```

### 2. Dependencias Instaladas
```bash
pip install langgraph langchain-tavily langchain-groq
```

### 3. API Keys Necesarias
```bash
# .env
GROQ_API_KEY=tu_key_aqui
TAVILY_API_KEY=tu_key_aqui
OPENAI_MODEL=llama-3.1-8b-instant
```

**Obtener Tavily API (gratis):** https://app.tavily.com/home

---

## 📚 Conceptos Clave

### 1. **ReAct (Reason + Act)**
Framework donde el agente:
1. **Piensa** (Reason): Analiza qué herramienta usar
2. **Actúa** (Act): Ejecuta la herramienta
3. **Observa**: Procesa el resultado
4. Repite hasta resolver la tarea

### 2. **LangGraph**
Framework para crear flujos de agentes:
- **Nodos**: Funciones que procesan el estado
- **Edges**: Conexiones entre nodos
- **State**: Diccionario compartido entre nodos

### 3. **Tavily Search**
API de búsqueda web optimizada para LLMs:
- Retorna resultados limpios y relevantes
- Configurable (depth, topic, time_range)
- Gratis hasta 1000 requests/mes

---

## 📝 Estructura del Notebook

### Celda 1-3: Setup
- Imports y configuración de Tavily

### Celda 4: Test ReAct Simple
- Crear agente ReAct con Tavily
- Test de búsqueda de trending topics

### Celda 7-10: Pipeline Completo con LangGraph
**Flujo:**
```
User Query → Refine Query → Search Topics → Analyze Sentiment → Result
```

**Componentes:**
1. `refine_user_query`: Mejora la consulta del usuario
2. `search_trending_topics`: Busca en web con ReAct agent
3. `analyze_sentiment`: Analiza sentimiento de resultados

---

## 🎓 Aprendizajes Clave

### ReAct vs RAG
| Aspecto | RAG | ReAct |
|---------|-----|-------|
| Datos | Estáticos (tu base de datos) | Dinámicos (web, APIs) |
| Actualización | Manual | Automática |
| Uso | Documentos propios | Información pública |

### Cuándo usar ReAct
- ✅ Necesitas información actualizada
- ✅ Datos cambian frecuentemente
- ✅ Consultas sobre noticias, precios, tendencias
- ❌ Información confidencial
- ❌ Respuestas instantáneas (más lento que RAG)

### Tavily vs Google Search
- **Tavily**: Optimizado para LLMs, resultados limpios
- **Google**: Más resultados, menos optimizado

---

## 📊 Ejemplo de Flujo

**Input:** "Is people buying ETH on the last month?"

**Paso 1 - Refine Query:**
```
"How many people bought Ethereum (ETH) in the last 30 days,
and what was the total ETH purchase volume?"
```

**Paso 2 - Search (ReAct Agent):**
```
Thought: Need to search for ETH buying trends
Action: tavily_search("ETH buyers last 30 days")
Observation: [Results from web]
```

**Paso 3 - Analyze Sentiment:**
```
"Positive - Rising on-chain activity, increased demand..."
```

---

## ✅ Checklist

- [x] Tavily API key obtenida
- [x] Dependencias instaladas (langgraph, langchain-tavily)
- [x] LLM adaptado a Groq
- [x] Tests exitosos
- [x] Documentación creada

---

## 🎯 Próximos Pasos

1. Ejecutar notebook celda por celda
2. Probar con tus propias queries
3. Experimentar con diferentes `search_depth` y `topic`
4. Combinar con RAG (agentic-rag.ipynb)

---

**Progreso:** 4/8 notebooks (50%)
