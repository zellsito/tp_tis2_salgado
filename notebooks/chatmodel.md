# Aprendizaje: chatmodel.ipynb

## 📋 Índice de Celdas

| # | Tipo | Descripción | ¿Ejecutar? |
|---|------|-------------|------------|
| 0 | Markdown | Título: "Prompting and basic Langchain" | No ejecutable |
| 1 | Markdown | Descripción de imports | No ejecutable |
| 2 | Code | Imports básicos (os, json, dotenv, etc.) | ✅ Ejecutar |
| 3 | Code | `load_dotenv()` - Cargar variables .env | ✅ Ejecutar |
| 4 | Code | `%pip install...` (COMENTADA) | ⏭️ Saltar |
| 5 | Code | Imports LangChain | ✅ Ejecutar |
| 6 | Markdown | Texto: "A simple LLM-based Chat" | No ejecutable |
| 7 | Code | Primera llamada a Groq (chiste) | ✅ Ejecutar |
| 8 | Markdown | Texto: "Chat models are based on roles" | No ejecutable |
| 9 | Code | Traducción con historial (chain) | ✅ Ejecutar |
| 10 | Markdown | Texto: "Zero-shot mode" | No ejecutable |
| 11 | Code | Zero-shot QA sobre Proxy pattern | ✅ Ejecutar |
| 12 | Markdown | Texto: "Handling Memory" | No ejecutable |
| 13 | Markdown | Texto: "Memory is a list of messages" | No ejecutable |
| 14 | Code | Contextualización de preguntas | ✅ Ejecutar |
| 15 | Markdown | Texto: "Use compressed history" | No ejecutable |
| 16 | Code | QA con memoria completa | ✅ Ejecutar |
| 17 | Code | `pprint.pprint(result)` | ✅ Ejecutar |
| 18 | Markdown | Texto: "Handling Few-shots" | No ejecutable |
| 19 | Code | Zero-shot antónimos | ✅ Ejecutar |
| 20 | Code | Definir ejemplos para few-shot | ✅ Ejecutar |
| 21 | Code | Selector semántico de ejemplos | ✅ Ejecutar |
| 22 | Markdown | Texto: "Use the new prompt in a chain" | No ejecutable |
| 23 | Code | Few-shot chain | ✅ Ejecutar |
| 24 | Markdown | Texto: "Generate response as JSON" | No ejecutable |
| 25 | Code | Structured output con Pydantic | ✅ Ejecutar |
| 26 | Code | `pprint.pprint(result.model_dump())` | ✅ Ejecutar |
| 27 | Markdown | Separador "---" | No ejecutable |

---

## 🔧 Errores Encontrados y Corregidos

### Error 1: Import deprecado (Celda 5)
**Problema:** `ChatMessageHistory` está deprecado
```python
# ❌ Versión antigua (da warning)
from langchain.memory import ChatMessageHistory

# ✅ Versión correcta
from langchain_core.chat_history import InMemoryChatMessageHistory
```

### Error 2: Modelo Groq deprecado (Celda 7)
**Problema:** `llama-3.1-70b-versatile` dado de baja por Groq
```bash
BadRequestError: The model `llama-3.1-70b-versatile` has been decommissioned
```

**Solución:** Actualizar `.env`
```bash
# ❌ Modelo antiguo
OPENAI_MODEL=llama-3.1-70b-versatile

# ✅ Modelo nuevo (más barato y rápido)
OPENAI_MODEL=llama-3.1-8b-instant
```

---

### Error 3: OpenAI Embeddings sin API Key (Celda 21)
**Problema:** `OpenAIEmbeddings()` requiere `OPENAI_API_KEY` pero estamos usando Groq
```python
OpenAIError: The api_key client option must be set either by passing api_key to the client
or by setting the OPENAI_API_KEY environment variable
```

**Causa:** Groq NO ofrece embeddings, solo modelos de chat (LLMs)

**Solución:** Usar embeddings gratuitos de HuggingFace (ver `../setup.md` para instalación)

**1. Actualizar Celda 5 (imports):**
```python
# ❌ Antiguo (OpenAI, de pago)
# from langchain_openai import OpenAIEmbeddings

# ✅ Nuevo (HuggingFace, gratis)
from langchain_huggingface import HuggingFaceEmbeddings
```

**2. Actualizar Celda 21 (selector semántico):**
```python
# ❌ Antiguo
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),  # Requiere OPENAI_API_KEY
    Chroma,
    k=1,
)

# ✅ Nuevo
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),  # Gratis, local
    Chroma,
    k=1,
)
```

**¿Qué hace `HuggingFaceEmbeddings`?**
- Descarga modelo `all-MiniLM-L6-v2` (pequeño, ~90MB)
- Convierte texto a vectores localmente (sin API, sin costo)
- Primera ejecución: descarga modelo, después usa caché
- 100% gratis, sin límites

---

## 📚 Conceptos Clave

### 1. Chaineo con Pipe `|`

El operador `|` conecta componentes en secuencia:

```python
chain = prompt | llm | parser
```

**Flujo:**
```
INPUT → prompt → llm → parser → OUTPUT
```

**Similar a TypeScript pipes:**
```typescript
// TypeScript (RxJS)
observable.pipe(map(...), filter(...))

// LangChain (Python)
chain = prompt | llm | parser
```

**Ejemplo:**
```python
# Sin pipe (manual)
formatted = prompt.format(input)
response = llm.invoke(formatted)
output = parser.parse(response)

# Con pipe (elegante)
chain = prompt | llm | parser
output = chain.invoke(input)
```

---

### 2. Variables en Templates: `{variable}` vs `input=`

#### ¿Qué es `{question}`?
Es un placeholder que se reemplaza al ejecutar:

```python
prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),  # ← Variable
])
```

#### Forma 1: Diccionario (explícito)
```python
result = chain.invoke({"question": "What is MVC?"})
```

#### Forma 2: Atajo `input=` (solo si hay 1 variable)
```python
result = chain.invoke(input="What is MVC?")
# ↑ Equivalente a {"question": "What is MVC?"}
```

#### Regla:
| Variables en template | Usar |
|----------------------|------|
| 1 variable | `input=` o diccionario |
| 2+ variables | Solo diccionario |

**Ejemplo con 2 variables:**
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    ("human", "{question}"),
])

# ❌ NO funciona
result = chain.invoke(input="What is MVC?")

# ✅ Funciona
result = chain.invoke({
    "role": "chef",
    "question": "How to make pasta?"
})
```

---

### 3. Tipos de Prompting

| Tipo | Descripción | Celdas |
|------|-------------|--------|
| **Zero-Shot** | Sin ejemplos, solo conocimiento del LLM | 11, 19 |
| **Few-Shot** | Con ejemplos para aprender el patrón | 20, 23 |
| **Semantic Few-Shot** | Ejemplos seleccionados por similitud | 21 |

---

### 4. Tipos de Templates

| Template | Para qué | Ejemplo |
|----------|----------|---------|
| `PromptTemplate` | Texto simple sin roles | Celda 19 |
| `ChatPromptTemplate` | Con roles (system/human/ai) | Celdas 9, 11 |
| `FewShotPromptTemplate` | Con ejemplos dinámicos | Celda 21 |

---

### 5. Memoria (Chat History)

```python
# Crear memoria
chat_history = InMemoryChatMessageHistory()

# Agregar mensajes
chat_history.add_user_message("¿Qué es MVC?")
chat_history.add_ai_message("MVC es un patrón...")

# Usar en siguiente pregunta
chain.invoke({
    "question": "¿Puedo combinarlo con otros?",
    "chat_history": chat_history.messages
})
```

**Problema que resuelve:** Preguntas ambiguas que necesitan contexto previo

---

### 6. Structured Output (Pydantic)

Forzar al LLM a devolver JSON con estructura específica:

```python
from pydantic import BaseModel, Field

class Antonym(BaseModel):
    antonym: str = Field(description="El antónimo")
    explanation: str = Field(description="Explicación")

llm_with_structure = llm.with_structured_output(Antonym)
result = chain.invoke(input="happy")

# result es un objeto Antonym, no texto
print(result.antonym)        # "sad"
print(result.explanation)    # "Happy means..."
```

---

## 📝 Explicación por Celda (Solo Código Ejecutable)

### Celda 2: Imports Básicos
```python
import os                              # Variables de entorno
from dotenv import load_dotenv         # Cargar .env
import pprint                          # Pretty print
from typing import List                # Type hints
from IPython.display import display, Markdown  # Mostrar en Jupyter
```

---

### Celda 3: Cargar .env
```python
load_dotenv()  # Retorna True si encontró .env
```
Lee `GROQ_API_KEY` y `OPENAI_MODEL` del archivo `.env`

---

### Celda 5: Imports LangChain
```python
from langchain_groq import ChatGroq                    # Cliente Groq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory  # ✅ CORREGIDO
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings       # ❌ OpenAI (de pago)
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ HuggingFace (gratis)
from pydantic import BaseModel, Field
```

---

### Celda 7: Primera Llamada a Groq
```python
llm_model = os.environ["OPENAI_MODEL"]  # llama-3.1-8b-instant
llm = ChatGroq(model=llm_model, temperature=0.1)

response = llm.invoke("Tell me a joke about data scientists")
print(response.content)
```

**Qué hace:**
- `temperature=0.1`: Poco creativo (determinista)
- `invoke()`: Llamada única al LLM

---

### Celda 9: Traducción con Historial
```python
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant..."),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | llm  # ← Chaineo

ai_msg = chain.invoke({
    "messages": [
        HumanMessage(content="Translate: I love programming."),
        AIMessage(content="J'adore la programmation."),
        HumanMessage(content="What did you just say?"),
    ],
})
print(ai_msg.content)
```

**Qué hace:**
- Simula conversación con historial
- El LLM ve mensajes previos y puede referirse a ellos

---

### Celda 11: Zero-Shot QA
```python
SYSTEM_PROMPT = "You are an experienced software architect..."

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

qa_chain = qa_prompt | llm

result = qa_chain.invoke(input="What are the pros and cons of the Proxy design pattern?")
display(Markdown(result.content))
```

**Qué hace:**
- QA sin ejemplos (zero-shot)
- `display(Markdown(...))`: Muestra con formato en Jupyter

---

### Celda 14: Contextualización de Preguntas
```python
CONTEXTUALIZED_PROMPT = """Given a chat history and the latest question
    which might reference context in the chat history, formulate a standalone question..."""

contextualized_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZED_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

contextualized_qa_chain = contextualized_qa_prompt | llm | StrOutputParser()

# Crear memoria
chat_history = InMemoryChatMessageHistory()
chat_history.add_user_message(query)   # Pregunta anterior
chat_history.add_ai_message(result)    # Respuesta anterior

# Nueva pregunta ambigua
query = "Can I combine the pattern with other patterns?"
ai_msg = contextualized_qa_chain.invoke({
    'question': query,
    'chat_history': chat_history.messages
})
print(ai_msg)
```

**Qué hace:**
- Reescribe preguntas ambiguas para que sean auto-contenidas
- Input: "Can I combine the pattern..." (¿cuál patrón?)
- Output: "Can I combine the Proxy design pattern..."

---

### Celda 16: QA con Memoria Completa
```python
def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualized_qa_chain
    else:
        return input["question"]

qa_chain_with_memory = (
    RunnablePassthrough.assign(
        context=contextualized_question | qa_prompt | llm
    )
)

result = qa_chain_with_memory.invoke({
    'question': query,
    'chat_history': chat_history.messages
})

display(Markdown(result['context'].content))
```

**Qué hace:**
- `RunnablePassthrough.assign()`: Pasa input original + agrega campo `context`
- Ejecuta sub-chain y guarda resultado en `context`

---

### Celda 19: Zero-Shot Antónimos
```python
zero_shot_prompt = PromptTemplate(
    input_variables=['input'],
    template="""Return the antonym of the input given along with an explanation.
    Input: {input}
    Output:
    Explanation:
    """
)

zero_shot_chain = zero_shot_prompt | llm
result = zero_shot_chain.invoke(input='I am very sad but still have hope')
print(result.content)
```

**Diferencia:** `PromptTemplate` (sin roles) vs `ChatPromptTemplate` (con roles)

---

### Celda 20: Definir Ejemplos Few-Shot
```python
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]
```

---

### Celda 21: Selector Semántico
```python
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    # OpenAIEmbeddings(),  # ❌ Antiguo: requiere OPENAI_API_KEY
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),  # ✅ Nuevo: gratis
    Chroma,              # Base de datos vectorial
    k=1,                 # Devolver 1 ejemplo más similar
)

similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Return the antonym...\n\nExample(s):",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

print(similar_prompt.format(adjective="rainy"))
```

**Qué hace:**
- Selecciona el ejemplo MÁS SIMILAR usando embeddings
- Usa modelo local `all-MiniLM-L6-v2` (primera vez descarga ~90MB)
- Para "rainy" → selecciona "sunny → gloomy"

---

### Celda 23: Few-Shot Chain
```python
few_shot_chain = similar_prompt | llm

query = 'rainy'
result = few_shot_chain.invoke(input=query)
print(result.content)
```

---

### Celda 25: Structured Output
```python
class FormattedAntonym(BaseModel):
    antonym: str = Field(description="An antonym for the input word or phrase.")
    explanation: str = Field(description="A short explanation...")
    additional_clarifications: List[str] = Field(description="...", default=[])

llm_with_structure = llm.with_structured_output(FormattedAntonym)

few_shot_chain1 = similar_prompt | llm_with_structure
result = few_shot_chain1.invoke(input=query)
result  # Es un objeto FormattedAntonym, no texto
```

**Resultado:**
```python
FormattedAntonym(
    antonym='sunny',
    explanation='Rainy weather has precipitation...',
    additional_clarifications=[]
)
```

---

### Celda 26: Convertir a Dict
```python
pprint.pprint(result.model_dump())
```

Convierte objeto Pydantic → diccionario Python

---

## 📊 Comportamiento de Jupyter Notebooks

### Al cerrar/abrir el notebook
- ❌ **Se pierde:** Variables, imports, objetos en memoria
- ✅ **Se guarda:** Código de las celdas

### Buenas prácticas
1. Siempre ejecutar de arriba hacia abajo
2. Usar "Run All" para ejecutar todo
3. El orden importa

### Numeración
- **Cell ID** (fijo): `5afca013`, `2172d63b` (usado internamente)
- **Número de ejecución** (dinámico): `[1]`, `[2]`, `[3]` (visible al ejecutar)

---

## 🎯 Resumen Ejecutivo

| Concepto | Qué es | Celda ejemplo |
|----------|--------|---------------|
| **Chaineo `\|`** | Conecta componentes en secuencia | 9, 11, 14 |
| **Variables `{var}`** | Placeholders en templates | 11, 19 |
| **Zero-Shot** | Sin ejemplos | 11, 19 |
| **Few-Shot** | Con ejemplos | 20, 23 |
| **Memoria** | Historial de conversación | 14, 16 |
| **Structured Output** | JSON validado con Pydantic | 25 |
| **StrOutputParser** | Extrae solo el string | 14 |
| **RunnablePassthrough** | Pasa input + agrega campos | 16 |

---

## 📚 Recursos Adicionales

### Documentación Oficial
- **LangChain:** https://python.langchain.com/docs/
- **Groq:** https://console.groq.com/docs/
- **HuggingFace:** https://huggingface.co/docs
- **Jupyter:** https://jupyter.org/documentation

### Modelos Recomendados
- **Groq (chat):** `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`
- **HuggingFace (embeddings):** `all-MiniLM-L6-v2`, `all-mpnet-base-v2`

### Próximos Pasos
1. Experimentar con diferentes modelos de Groq
2. Modificar prompts para obtener diferentes resultados
3. Crear tus propios ejemplos few-shot
4. Explorar RAG (Retrieval-Augmented Generation)

---

## ✅ Checklist de Ejecución

- [ ] Entorno configurado (ver `setup.md`)
- [ ] Kernel reiniciado
- [ ] Celdas 2-5 ejecutadas (setup)
- [ ] Celda 7 ejecutada (primera llamada a Groq funciona)
- [ ] Celda 9 ejecutada (traducción con historial)
- [ ] Celda 11 ejecutada (zero-shot QA)
- [ ] Celda 14 ejecutada (contextualización)
- [ ] Celda 21 ejecutada (embeddings locales descargados)
- [ ] Celda 23 ejecutada (few-shot)
- [ ] Celda 25 ejecutada (structured output)

---

## 🎓 Aprendizajes Clave

### Sobre Jupyter Notebooks
- Los notebooks combinan código, texto y visualizaciones
- El kernel se reinicia al cerrar → re-ejecutar celdas al abrir
- Ejecutar siempre de arriba hacia abajo para evitar errores
- Usar "Run All" para ejecutar todo de una vez

### Sobre LangChain
- El pipe `|` conecta componentes en secuencia
- Templates con `{variables}` permiten reutilizar prompts
- Memory permite que el LLM recuerde contexto
- Structured Output valida respuestas con Pydantic

### Sobre LLMs
- Temperature controla creatividad (0.1 = determinista, 1.0 = creativo)
- Zero-shot = sin ejemplos, Few-shot = con ejemplos
- Groq ofrece LLMs gratuitos (solo chat, no embeddings)
- HuggingFace ofrece embeddings gratuitos (locales)

---

## 📝 Notas Finales

**Este documento está completo.** Contiene:
- ✅ Todas las celdas explicadas
- ✅ Errores encontrados y soluciones
- ✅ Conceptos clave con ejemplos
- ✅ Comparaciones con TypeScript
- ✅ Resumen ejecutivo

Para configurar el entorno desde cero, revisar `setup.md`.
