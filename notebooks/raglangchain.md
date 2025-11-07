# Aprendizaje: raglangchain.ipynb

## 📋 Resumen Ejecutivo

**Notebook:** raglangchain.ipynb (254KB adaptado)
**Tema principal:** Retrieval Augmented Generation (RAG) - Sistema que combina búsqueda semántica con LLMs
**Estado:** ✅ COMPLETADO - 100% funcional con embeddings locales
**Fecha:** 2025-11-07

### 🎯 Objetivo del Notebook
Aprender RAG (Retrieval Augmented Generation), una técnica que combina:
1. **Búsqueda semántica** (recuperar documentos relevantes de una base de datos vectorial)
2. **LLM** (generar respuestas basadas en los documentos recuperados)

RAG permite que un LLM responda preguntas con información actualizada y específica de tus documentos, sin necesidad de re-entrenar el modelo.

---

## 🔧 Errores Encontrados y Corregidos

### 1. Imports Deprecados de LangChain 1.0+
```python
# ❌ ANTES (LangChain 0.x)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma

# ✅ DESPUÉS (LangChain 1.0+)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
```

### 2. Hub de LangChain
```python
# ❌ ANTES
from langchain import hub
rag_prompt = hub.pull("rlm/rag-prompt")

# ✅ DESPUÉS
from langsmith import Client as LangSmithClient
hub_client = LangSmithClient()
rag_prompt = hub_client.pull_prompt("rlm/rag-prompt")
```

### 3. Embeddings y LLM (OpenAI → Local/Gratis)
```python
# ❌ ANTES
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

# ✅ DESPUÉS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
```

### 4. Re-ranking (ContextualCompressionRetriever → Manual)
```python
# ❌ ANTES (No disponible en LangChain 1.0+)
from langchain.retrievers import ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(...)

# ✅ DESPUÉS (Implementación manual)
from sentence_transformers import CrossEncoder

cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query: str, documents: list, top_n: int = 3):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder_model.predict(pairs)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_n]]
```

### 5. RAGxplorer (Instalación y Configuración)
```bash
# ❌ pip install ragexplorer  # No disponible en PyPI
# ✅ Instalar desde GitHub
pip install git+https://github.com/gabrielchua/RAGxplorer.git

# Parches necesarios:
# 1. ragxplorer/rag.py - actualizar imports de text_splitter
# 2. ragxplorer/projections.py - convertir listas a numpy arrays
```

```python
# ✅ Uso con embeddings locales
from ragxplorer import RAGxplorer

client = RAGxplorer(embedding_model="all-MiniLM-L6-v2")
client.load_pdf(document_path="./data/Understanding_Climate_Change.pdf",
                chunk_size=1000, chunk_overlap=100)

# Usar método "naive" en lugar de "HyDE" (bug con embeddings locales)
client.visualize_query(query="What is climate change?",
                       retrieval_method="naive", top_k=6)
```

---

## 📚 Conceptos Clave

### 1. **RAG (Retrieval Augmented Generation)**
Combina dos componentes:
- **Retrieval:** Busca documentos relevantes en una base de datos vectorial
- **Generation:** LLM genera respuesta usando los documentos como contexto

**Flujo:**
```
User Query → Vector Search → Top K Docs → (Optional Re-ranking) → LLM → Answer
```

### 2. **Embeddings**
Representación vectorial del texto que captura su significado semántico.
- Usamos `all-MiniLM-L6-v2` (384 dimensiones, local, gratis)
- Textos similares tienen embeddings cercanos en el espacio vectorial

### 3. **Vector Store (ChromaDB)**
Base de datos que almacena documentos como vectores.
- Permite búsqueda por similitud semántica
- En este notebook: ChromaDB con persistencia en memoria

### 4. **Chunking (División de Documentos)**
Dividir documentos largos en fragmentos pequeños para:
- Mejorar la relevancia de la búsqueda
- Caber en el contexto del LLM
- `chunk_size=1000`, `chunk_overlap=100`

### 5. **Re-ranking con Cross-Encoder**
Mejora la calidad de los documentos recuperados:
- **Bi-encoder** (embeddings): Rápido, busca en millones de docs
- **Cross-encoder**: Lento pero más preciso, re-rankea top K

**Diferencia:**
- Bi-encoder: Encode query y docs por separado, compara vectores
- Cross-encoder: Encode (query + doc) juntos, score de relevancia directo

### 6. **Prompt Template para RAG**
```
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.

Context: {context}
Question: {question}

Answer:
```

---

## 📝 Estructura del Notebook

### Parte 1: Setup y Búsqueda Semántica con Movies
- Cargar dataset de películas (JSON)
- Crear documentos con metadata (género, fecha, idioma)
- Vector store en memoria (`InMemoryVectorStore`)
- Búsqueda semántica básica con scores
- Crear retriever con filtros

**Conceptos aprendidos:**
- `similarity_search()` vs `similarity_search_with_score()`
- Filtros con metadata (ej: `genre == 'Horror'`)
- Retrievers como abstracción sobre vector stores

### Parte 2: RAG Chain Básico
- Configurar LLM (ChatGroq)
- Obtener prompt de LangSmith Hub
- Crear chain con LCEL (LangChain Expression Language)
- Formato: `{"context": retriever, "question": input} | prompt | llm | parser`

**Ejemplo de query:**
```python
query = "I want to get a movie about religion"
result = rag_chain.invoke(query)
# Retriever busca películas relevantes → LLM responde basándose en ellas
```

### Parte 3: RAG con PDF (Climate Change)
- Cargar PDF con `PyPDFLoader`
- Chunking con `RecursiveCharacterTextSplitter`
- Crear ChromaDB con 97 chunks
- Retriever con k=5 documentos
- Pretty print de documentos recuperados

**Resultado:** Sistema que responde preguntas sobre cambio climático usando el PDF

### Parte 4: Re-ranking
- Implementación manual con `CrossEncoder`
- Función `rerank_documents()` que:
  1. Obtiene top K docs del retriever (ej: k=5)
  2. Re-rankea con cross-encoder
  3. Retorna top N mejores (ej: n=3)
- Clase `RerankedRetriever` para integrar en chains

**Mejora observable:** Documentos más relevantes en top positions

### Parte 5: RAGxplorer (Visualización)
- Carga PDF y crea vector database
- Reduce dimensionalidad con UMAP (384D → 2D)
- Visualiza chunks y query en espacio 2D
- Marca documentos recuperados en verde

**Utilidad:** Ver visualmente qué chunks están cerca de la query

---

## 🎓 Aprendizajes Clave

### 1. **RAG es más útil que fine-tuning para datos específicos**
- No requiere re-entrenar el modelo
- Fácil actualizar información (agregar/quitar docs)
- El LLM puede citar fuentes específicas

### 2. **Pipeline típico de RAG**
```python
# 1. Cargar documentos
docs = PyPDFLoader("file.pdf").load()

# 2. Chunking
chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(docs)

# 3. Crear vector store
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 5. RAG Chain
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | parser
```

### 3. **Trade-offs importantes**
- **Chunk size:** Pequeño = preciso pero fragmentado, Grande = contexto completo pero menos específico
- **Top K:** Más docs = más contexto pero más ruido
- **Re-ranking:** Mejora calidad pero agrega latencia

### 4. **LCEL (LangChain Expression Language)**
Sintaxis con pipe `|` para encadenar componentes:
```python
chain = step1 | step2 | step3
result = chain.invoke(input)
```

Equivalente a:
```python
temp1 = step1(input)
temp2 = step2(temp1)
result = step3(temp2)
```

### 5. **Metadata es poderosa**
Permite filtros avanzados:
```python
# Solo películas de horror en 2023
retriever.invoke(query, filter={
    "genre": "Horror",
    "release_date": {"$gte": "2023-01-01"}
})
```

---

## 🔍 Comparación: Sin Re-ranking vs Con Re-ranking

### Sin Re-ranking (solo embeddings)
```
Query: "What is the main cause of climate change?"

Top 3 documentos:
1. ...Holocene epoch...greenhouse gases...
2. ...agricultural sector's carbon footprint...
3. ...Understanding Climate Change Chapter 1...
```

### Con Re-ranking (cross-encoder)
```
Query: "What is the main cause of climate change?"

Re-ranking scores:
1. Score:  5.4590 - ...greenhouse gases...primary cause...
2. Score:  1.3634 - ...Introduction to Climate Change...
3. Score: -0.4736 - ...agricultural sector...

✅ Documentos re-ordenados por relevancia semántica más precisa
```

**Observación:** El cross-encoder detecta que el primer documento es MÁS relevante porque menciona directamente "primary cause" y "greenhouse gases".

---

## 📊 Métricas del Notebook

- **PDF:** Understanding_Climate_Change.pdf (33 páginas, 206KB)
- **Chunks generados:** 97
- **Embedding model:** all-MiniLM-L6-v2 (384 dimensiones, ~90MB)
- **LLM:** llama-3.1-8b-instant (Groq, gratis)
- **Cross-encoder:** ms-marco-MiniLM-L-6-v2 (~80MB)
- **Dataset movies:** 10 películas con metadata

---

## ✅ Checklist de Ejecución

- [x] Instalar dependencias (pypdf, langsmith, ragexplorer)
- [x] Adaptar imports para LangChain 1.0+
- [x] Configurar embeddings locales (HuggingFace)
- [x] Configurar LLM gratuito (Groq)
- [x] Copiar PDF a `notebooks/data/`
- [x] Ejecutar búsqueda semántica con movies
- [x] Crear RAG chain básico
- [x] Cargar y procesar PDF (chunking)
- [x] Implementar re-ranking manual
- [x] Configurar RAGxplorer
- [x] Probar visualizaciones
- [x] Documentar en `raglangchain.md`

---

## 📝 Notas Finales

### Diferencias con chatmodel.ipynb
- **chatmodel:** Prompting, chains, memoria, few-shot
- **raglangchain:** Vector search + LLM, retrieval, re-ranking

### Diferencias con semanticsearchnotebook.ipynb
- **semanticsearch:** Solo búsqueda vectorial (embeddings + ChromaDB)
- **raglangchain:** Búsqueda vectorial + LLM para responder

### Próximos pasos recomendados
1. **Ejecutar celda por celda** y observar resultados
2. **Experimentar con chunk_size** (500, 1000, 2000)
3. **Probar diferentes queries** en el PDF
4. **Comparar con/sin re-ranking** en tus propios datos
5. **Continuar con:** `react-web-search.ipynb` (agentes con búsqueda web)

---

## 🎯 Conceptos para Recordar

| Concepto | Descripción | Ejemplo |
|----------|-------------|---------|
| **RAG** | Retrieval + Generation | Buscar docs → LLM responde |
| **Embeddings** | Vectores semánticos | "perro" ≈ "can" |
| **Chunking** | Dividir docs en fragmentos | PDF → 97 chunks de 1000 chars |
| **Retriever** | Busca top K docs | k=5 documentos más similares |
| **Re-ranking** | Mejora orden de docs | Cross-encoder scores |
| **Vector Store** | DB de embeddings | ChromaDB, FAISS, Pinecone |
| **LCEL** | Chain con pipes | `step1 \| step2 \| step3` |

---

**🎓 Conclusión:**
RAG es una técnica fundamental para crear LLMs que usen información específica de tu dominio sin fine-tuning. Este notebook cubre todo el pipeline: desde cargar documentos hasta generar respuestas con re-ranking y visualización.

**Progreso:** 3/8 notebooks completados (37.5%) ✅
