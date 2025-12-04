# Prompt de Contexto - Proyecto TIS2 Aprendizaje Notebooks

## 🎯 Objetivo del Proyecto

Aprender Jupyter Notebooks + Python + LangChain ejecutando y documentando notebooks paso a paso.

---

## 📋 Metodología de Trabajo

### Para cada notebook:
1. **Ejecutar celda por celda** (con Groq API gratuita)
2. **Documentar en archivo `.md`** con el mismo nombre (ej: `chatmodel.ipynb` → `chatmodel.md`)
3. **Actualizar `notebooks/README.md`** agregando el nuevo notebook a la lista
4. **Mantener estructura organizada**

---

## 📂 Estructura del Proyecto

```
tp_tis2_salgado/
├── README.md                    # Proyecto principal (pendiente implementación)
├── setup.md                     # Setup completo del entorno (Python, Groq, HuggingFace)
├── .env                         # GROQ_API_KEY + OPENAI_MODEL=llama-3.1-8b-instant
├── .gitignore                   # Archivos ignorados
├── .venv/                       # Entorno virtual Python 3.11.2
└── notebooks/
    ├── README.md                # Índice de todos los notebooks aprendidos
    ├── chatmodel.ipynb          # ✅ COMPLETADO
    ├── chatmodel.md             # ✅ Documentación completa
    ├── semanticsearchnotebook.ipynb  # 📌 SIGUIENTE
    └── (otros notebooks...)
```

---

## ✅ Notebooks Completados

### 1. chatmodel.ipynb (COMPLETADO)
- **Tamaño:** 13KB
- **Temas:** Prompting básico, chains, memoria, few-shot, structured output
- **Conceptos clave:**
  - Chaineo con pipe `|`
  - Templates con `{variables}`
  - Memoria (`InMemoryChatMessageHistory`)
  - Embeddings locales (`HuggingFaceEmbeddings`)
  - Zero-shot vs Few-shot
  - Structured output con Pydantic
- **Archivo de documentación:** `notebooks/chatmodel.md` (completo)

---

## 📌 Siguiente Notebook Recomendado

### semanticsearchnotebook.ipynb
**Por qué este?**
- ✅ Tamaño pequeño (36KB) - fácil de completar
- ✅ Relacionado con `chatmodel.ipynb` (usa embeddings)
- ✅ Introduce búsqueda semántica (concepto clave para RAG)
- ✅ Usa ChromaDB (ya instalado)
- ✅ Complejidad baja-media

**Temas que cubre:**
- Embeddings y vectorización
- Búsqueda semántica
- Base de datos vectorial (Chroma)
- Similarity search

**Siguiente después:** `raglangchain.ipynb` (RAG básico)

---

## 📊 Inventario Completo de Notebooks (11 total)

### 🎯 Notebooks de LLM/LangChain (Prioridad Alta - 8 notebooks)

| # | Notebook | Tamaño | Estado | Complejidad | Temas |
|---|----------|--------|--------|-------------|-------|
| 1 | chatmodel.ipynb | 13KB | ✅ COMPLETADO | Baja | Prompting, chains, memoria |
| 2 | semanticsearchnotebook.ipynb | 32KB | ✅ COMPLETADO | Baja-Media | Embeddings, búsqueda semántica |
| 3 | raglangchain.ipynb | 33KB | ✅ COMPLETADO | Media | RAG + re-ranking |
| 4 | react-web-search.ipynb | 49KB | ✅ COMPLETADO | Media | ReAct agents + Tavily |
| 5 | agentic-rag.ipynb | 393KB | ✅ COMPLETADO | Alta | RAG con flujo decisiones |
| 6 | raglangchaimongodb.ipynb | 263KB | 📌 SIGUIENTE | Media-Alta | RAG + MongoDB |
| 7 | sql-agent.ipynb | 1.1MB | ⏳ PENDIENTE | Alta | Agentes + SQL + LangGraph |
| 8 | langchainmultiagentcollaboration.ipynb | 1.1MB | ⏳ PENDIENTE | Muy Alta | Multi-agentes colaborativos |

### 📊 Notebooks de ML/Data Science (Prioridad Baja - 3 notebooks)

| # | Notebook | Tamaño | Estado | Tema |
|---|----------|--------|--------|------|
| 9 | pneumoniapreprocessing.ipynb | 47KB | ⏸️ OPCIONAL | Preprocesamiento imágenes |
| 10 | salarypredictionregression.ipynb | 144KB | ⏸️ OPCIONAL | Regresión ML |
| 11 | customerchurnclassification-fs.ipynb | 529KB | ⏸️ OPCIONAL | Clasificación ML |

**Progreso LLM/LangChain:** 5/8 completados (62.5%)
**Progreso Total:** 5/11 notebooks (45.5%)

---

## 🛠️ Configuración Actual

### Entorno
- Python 3.11.2
- Entorno virtual `.venv` activo
- VS Code con extensión Jupyter

### Dependencias Instaladas
```bash
# LLM & LangChain
langchain
langchain-core
langchain-community
langchain-groq
langchain-openai
langchain-chroma

# Embeddings locales
sentence-transformers
langchain-huggingface

# Utilidades
python-dotenv
jupyter

# Total: ~3GB
```

### API Keys (.env)
```bash
GROQ_API_KEY=***REMOVED***
OPENAI_MODEL=llama-3.1-8b-instant
```

---

## 📝 Formato de Documentación (template)

Para cada notebook crear archivo `<nombre>.md` con:

```markdown
# Aprendizaje: <nombre>.ipynb

## 📋 Índice de Celdas
(Tabla con todas las celdas numeradas)

## 🔧 Errores Encontrados y Corregidos
(Si aplica)

## 📚 Conceptos Clave
(Explicación de conceptos principales)

## 📝 Explicación por Celda
(Solo celdas ejecutables)

## 🎯 Resumen Ejecutivo
(Tabla de conceptos)

## ✅ Checklist de Ejecución

## 🎓 Aprendizajes Clave

## 📝 Notas Finales
```

---

## 🔄 Flujo de Trabajo

### Al empezar un nuevo notebook:

1. **Abrir notebook** en VS Code
2. **Seleccionar kernel** `.venv/bin/python`
3. **Ejecutar celdas** de arriba hacia abajo
4. **Anotar errores** y soluciones
5. **Crear archivo `.md`** con documentación completa
6. **Actualizar `notebooks/README.md`** agregando entrada del nuevo notebook
7. **Verificar** que todo funcione
8. **(Opcional) Limpiar contexto** de Claude Code para ahorrar tokens

---

## 📖 Archivos de Referencia

- **`setup.md`** - Setup completo (consultar si hay errores de instalación)
- **`notebooks/chatmodel.md`** - Ejemplo de documentación completa
- **`notebooks/README.md`** - Índice de notebooks aprendidos

---

## 🚨 Correcciones Aplicadas (para referencia futura)

### chatmodel.ipynb
1. **Import deprecado:**
   - ❌ `from langchain.memory import ChatMessageHistory`
   - ✅ `from langchain_core.chat_history import InMemoryChatMessageHistory`

2. **Modelo Groq deprecado:**
   - ❌ `llama-3.1-70b-versatile`
   - ✅ `llama-3.1-8b-instant`

3. **Embeddings (OpenAI → HuggingFace):**
   - ❌ `from langchain_openai import OpenAIEmbeddings`
   - ✅ `from langchain_huggingface import HuggingFaceEmbeddings`
   - ❌ `OpenAIEmbeddings()`
   - ✅ `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`

### raglangchain.ipynb (ADAPTADO ✅)
1. **Celda 0359a684 (imports - TODOS actualizados para LangChain 1.0+):**
   - ❌ `from langchain.document_loaders import PyPDFLoader`
   - ✅ `from langchain_community.document_loaders import PyPDFLoader`
   - ❌ `from langchain.text_splitter import RecursiveCharacterTextSplitter`
   - ✅ `from langchain_text_splitters import RecursiveCharacterTextSplitter`
   - ❌ `from langchain.schema import Document`
   - ✅ `from langchain_core.documents import Document`
   - ❌ `from langchain.vectorstores.chroma import Chroma`
   - ✅ `from langchain_chroma import Chroma`
   - ❌ `from langchain.schema.runnable import RunnablePassthrough`
   - ✅ `from langchain_core.runnables import RunnablePassthrough`
   - ❌ `from langchain.schema.output_parser import StrOutputParser`
   - ✅ `from langchain_core.output_parsers import StrOutputParser`
   - ❌ `from langchain import hub`
   - ✅ `from langsmith import Client as LangSmithClient` + `hub_client = LangSmithClient()`
   - ❌ `from langchain_openai import OpenAIEmbeddings, ChatOpenAI`
   - ✅ `from langchain_huggingface import HuggingFaceEmbeddings`
   - ✅ `from langchain_groq import ChatGroq`
2. **Celda 8o9x9mda5pj (nueva, configuración embeddings):**
   - ✅ `EMBEDDING_MODEL = "all-MiniLM-L6-v2"`
   - ✅ `embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)`
3. **Celda 90cab636 (dataset path):**
   - ❌ `input_datapath = "../semantic-search/dataset.json"`
   - ✅ `input_datapath = "dataset.json"`
4. **Celda 69dd1aea (InMemoryVectorStore):**
   - ❌ `InMemoryVectorStore(OpenAIEmbeddings())`
   - ✅ `InMemoryVectorStore(embeddings)`
5. **Celda 964b9696 (LLM):**
   - ❌ `ChatOpenAI(model=llm_model, temperature=0.1)`
   - ✅ `ChatGroq(model=llm_model, temperature=0.1)`
6. **Celda 9f41466a (hub.pull):**
   - ❌ `rag_prompt = hub.pull("rlm/rag-prompt")`
   - ✅ `rag_prompt = hub_client.pull_prompt("rlm/rag-prompt")`
7. **Celda 1779f900 (Chroma):**
   - ❌ `Chroma.from_documents(cleaned_texts, OpenAIEmbeddings())`
   - ✅ `Chroma.from_documents(cleaned_texts, embeddings)`
8. **Celdas 3652ba2b, 7250b7ca, 1316d1f7 (Re-ranking - COMPLETADO ✅):**
   - ✅ Implementado re-ranking manual con `CrossEncoder`
   - ✅ Función `rerank_documents()` creada
   - ✅ Clase `RerankedRetriever` para integrar re-ranking en chains
   - ✅ RAG chain con re-ranking funcionando correctamente
9. **Datos preparados:**
   - ✅ PDF copiado a `notebooks/data/Understanding_Climate_Change.pdf`
   - ✅ Dataset de películas en mismo directorio

### raglangchain.ipynb - Errores Corregidos (COMPLETO ✅)

**Error 1: Múltiples imports deprecados (LangChain 1.0+)**
```python
# ❌ Errores: No module named 'langchain.schema', 'langchain.retrievers', etc.

# Causa: LangChain 1.0+ reorganizó todos los módulos en paquetes separados

# ✅ Solución: Usar imports específicos de cada paquete
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
```

**Error 2: hub.pull() no disponible**
```python
# ❌ Error: cannot import name 'hub' from 'langchain'
# ❌ langchainhub está deprecado

# ✅ Solución: Usar langsmith Client (langsmith ya instalado)
from langsmith import Client as LangSmithClient
hub_client = LangSmithClient()
rag_prompt = hub_client.pull_prompt("rlm/rag-prompt")
```

**Error 3: ContextualCompressionRetriever no disponible (SOLUCIONADO ✅)**
```python
# ❌ Error: No module named 'langchain.retrievers'
# ❌ ContextualCompressionRetriever removido en LangChain 1.0+

# ✅ Solución: Implementar re-ranking manual con CrossEncoder
from sentence_transformers import CrossEncoder

cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query: str, documents: list, top_n: int = 3):
    """Re-rankea documentos usando cross-encoder"""
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder_model.predict(pairs)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_n]]

# Retriever personalizado con re-ranking integrado
class RerankedRetriever:
    def __init__(self, base_retriever, rerank_function, top_n=3):
        self.base_retriever = base_retriever
        self.rerank_function = rerank_function
        self.top_n = top_n

    def invoke(self, query: str):
        docs = self.base_retriever.invoke(query)
        return self.rerank_function(query, docs, self.top_n)
```

**Error 4: OpenAI API key no configurada**
```python
# ✅ Solución: Usar alternativas gratuitas
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
```

**Error 5: Path incorrecto del dataset**
```python
# ❌ input_datapath = "../semantic-search/dataset.json"
# ✅ input_datapath = "dataset.json"
```

**Error 6: RAGxplorer no instalado y con imports deprecados**
```python
# ❌ Error: No module named 'ragexplorer'
# Causa: No está disponible en PyPI normalmente

# ✅ Solución: Instalar desde GitHub
pip install git+https://github.com/gabrielchua/RAGxplorer.git

# ✅ Parche imports deprecados en ragxplorer/rag.py:
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# ✅ Parche bug en ragxplorer/projections.py (línea 47):
if isinstance(embedding, list):
    embedding = np.array(embedding)
```

**Error 7: HyDE retrieval method con bug en RAGxplorer**
```python
# ❌ retrieval_method="HyDE" causa AttributeError con embeddings locales
# ✅ Usar retrieval_method="naive" (método básico funciona correctamente)
```

**Celda 64463bd5 (instalación):**
- ✅ `ragexplorer` instalado desde GitHub
- ✅ `nbformat` ya instalado

**Celda 367b91c6 (inicialización):**
- ❌ `RAGxplorer(embedding_model="text-embedding-3-small")` (OpenAI)
- ✅ `RAGxplorer(embedding_model="all-MiniLM-L6-v2")` (local, gratis)

**Celda 4d895962 (visualización):**
- ❌ `retrieval_method="HyDE"` (bug con embeddings locales)
- ✅ `retrieval_method="naive"` (método básico, funciona bien)

---

### semanticsearchnotebook.ipynb (ADAPTADO ✅)
1. **Celda 482c51f4:** Comentada instalación de OpenAI, todo ya instalado en .venv
2. **Celda 3a43fb47 (imports):**
   - ❌ `import openai`
   - ❌ `from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction`
   - ✅ `from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction`
3. **Celda a3fbd451 (setup):**
   - ❌ Todo bloque OpenAI API key comentado
   - ✅ `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` (modelo local)
4. **Celda cd722937 (ChromaDB):**
   - ❌ `OpenAIEmbeddingFunction(api_key=..., model_name=...)`
   - ✅ `SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)`
5. **Dataset limpiado:**
   - ❌ `dataset.json` tenía embeddings OpenAI pre-calculados (dimensión 1536)
   - ✅ Backup creado: `dataset_original_with_openai_embeddings.json` (210KB)
   - ✅ `dataset.json` nuevo (8.6KB) sin embeddings, ChromaDB los genera localmente
6. **Celda d3743ae9 (Create Collection):**
   - ❌ `embeddings=df.embedding.tolist()` (usaba embeddings pre-calculados)
   - ✅ `documents=documents` (ChromaDB genera embeddings automáticamente)
7. **Celda 50ce16b3 (Preview):**
   - ❌ Esperaba embeddings en dataset
   - ✅ Indica que ChromaDB generará embeddings localmente

---

## 💡 Tips de Trabajo

### Jupyter Notebooks
- El kernel se reinicia al cerrar → re-ejecutar celdas al abrir
- Siempre ejecutar de arriba hacia abajo
- Usar "Run All" para ejecutar todo de una vez

### Claude Code
- Cuando el contexto sea muy largo (>100k tokens), limpiarlo:
  1. Guardar este archivo `PROMPT_CONTEXTO.md`
  2. Copiar contenido
  3. Reiniciar sesión de Claude Code
  4. Pegar el prompt para retomar

### Documentación
- Ser conciso pero completo
- Incluir ejemplos de código
- Explicar el "por qué", no solo el "qué"
- Agregar comparaciones con otros lenguajes si ayuda (ej: TypeScript)

---

## 🎯 Próximo Paso Inmediato

**Ejecutar y documentar: `semanticsearchnotebook.ipynb`**

1. Abrir `notebooks/semanticsearchnotebook.ipynb`
2. Seleccionar kernel `.venv/bin/python`
3. Ejecutar celdas en orden
4. Crear `notebooks/semanticsearchnotebook.md`
5. Actualizar `notebooks/README.md`

---

## 📞 Forma de Trabajo con Claude Code

**Instrucciones claras:**
- "Ejecuta la celda X y documenta qué hace"
- "Hay un error en la celda Y, corrigelo y documenta la solución"
- "Actualiza notebooks/README.md agregando semanticsearchnotebook"
- "Muéstrame un resumen de lo aprendido en esta sesión"

**Lo que Claude Code debe hacer:**
- Leer notebooks
- Ejecutar código (cuando sea posible)
- Documentar explicaciones
- Corregir errores
- Mantener archivos actualizados
- Ser conciso pero completo

---

## ✅ Estado Actual

- ✅ Setup completo
- ✅ `chatmodel.ipynb` completado y documentado
- ✅ `semanticsearchnotebook.ipynb` completado y documentado
- ✅ `raglangchain.ipynb` completado y documentado
  - ✅ Dependencias instaladas (pypdf, langsmith, ragexplorer)
  - ✅ Notebook adaptado (OpenAI → Groq + HuggingFace)
  - ✅ PDF preparado en notebooks/data/
  - ✅ RAGxplorer configurado con embeddings locales
  - ✅ Todos los imports actualizados para LangChain 1.0+
  - ✅ Re-ranking implementado con CrossEncoder (ms-marco-MiniLM-L-6-v2)
  - ✅ Visualización RAGxplorer funcionando
- ✅ `react-web-search.ipynb` adaptado y documentado
  - ✅ Tavily API key configurada
  - ✅ Dependencias instaladas (langgraph, langchain-tavily)
  - ✅ LLM adaptado (OpenAI → Groq)
  - ✅ Tests exitosos
- ✅ `agentic-rag.ipynb` completado y documentado
  - ✅ Todos los imports actualizados a LangChain 1.0+
  - ✅ LLM adaptado (OpenAI → Groq)
  - ✅ Embeddings adaptados (OpenAI → HuggingFace local)
  - ✅ LangGraph workflow con nodos de decisión inteligentes
  - ✅ PDF de ejemplo disponible (Understanding_Climate_Change.pdf)
  - ✅ Flujo completo: query generation → retrieval → grading → rewriting → answer
- 🎯 **Objetivo:** Aprender 8 notebooks de LLM/LangChain (+ 3 opcionales de ML)
- 📝 **Progreso LLM:** 5/8 completado (62.5%)
- 📝 **Progreso Total:** 5/11 notebooks (45.5%)

---

## 🎯 Próximos Pasos Recomendados

### Opción A: raglangchaimongodb.ipynb ⭐ RECOMENDADA
**Por qué seguir con este:**
- ✅ Continuidad lógica después de RAG básico + agentic-rag
- ✅ Introduce persistencia con MongoDB (crucial para apps reales)
- ✅ Tamaño mediano (263KB) - abordable después de agentic-rag (393KB)
- ✅ Complejidad Media-Alta (desafiante pero no abrumador)
- ✅ Combina RAG + Base de datos vectorial + Filtros tradicionales

**Temas que aprenderás:**
- MongoDB Atlas como vector store
- Persistencia de embeddings en BD NoSQL
- Queries híbridas (vectorial + metadatos)
- Integración LangChain + PyMongo

### Opción B: sql-agent.ipynb
**Consideraciones:**
- ⚠️ 1.1MB (muy extenso, >3x más grande que mongodb)
- ⚠️ Complejidad Alta
- ✅ Prerequisitos cumplidos (react-web-search + agentic-rag)
- Recomendación: Dejar para después de mongodb

### Opción C: Notebooks ML/Data Science
- Solo si querés cambiar de tema temporalmente
- Menor prioridad para LangChain/LLM

---

**Última actualización:** 2025-11-07
**Sesión actual:** agentic-rag.ipynb COMPLETADO ✅
**Siguiente recomendado:** raglangchaimongodb.ipynb (RAG + MongoDB, 263KB)
**Notebooks pendientes LLM:** 3 (mongodb, sql-agent, multiagent)
