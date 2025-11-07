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

## 🔄 Orden de Aprendizaje Recomendado (por complejidad)

| Orden | Notebook | Tamaño | Complejidad | Temas |
|-------|----------|--------|-------------|-------|
| 1 | ✅ `chatmodel.ipynb` | 13KB | Baja | Prompting, chains, memoria |
| 2 | 📌 `semanticsearchnotebook.ipynb` | 36KB | Baja-Media | Embeddings, búsqueda semántica |
| 3 | `react-web-search.ipynb` | 78KB | Media | ReAct agents, web search |
| 4 | `raglangchain.ipynb` | 254KB | Media | RAG básico |
| 5 | `raglangchaimongodb.ipynb` | 258KB | Media-Alta | RAG + MongoDB |
| 6 | `agentic-rag.ipynb` | 385KB | Alta | RAG con agentes |
| 7 | `sql-agent.ipynb` | 1.1MB | Alta | Agentes SQL |
| 8 | `langchainmultiagentcollaboration.ipynb` | 1.1MB | Muy Alta | Multi-agentes |

**Notebooks de ML/Data Science (opcionales, menor prioridad):**
- `salarypredictionregression.ipynb` (141KB)
- `customerchurnclassification-fs.ipynb` (517KB)
- `pneumoniapreprocessing.ipynb` (46KB)

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
1. **Celda 0359a684 (imports):**
   - ❌ `from langchain_openai import OpenAIEmbeddings`
   - ❌ `from langchain_openai import ChatOpenAI`
   - ✅ `from langchain_huggingface import HuggingFaceEmbeddings`
   - ✅ `from langchain_groq import ChatGroq`
2. **Celda 8o9x9mda5pj (nueva, configuración embeddings):**
   - ✅ `EMBEDDING_MODEL = "all-MiniLM-L6-v2"`
   - ✅ `embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)`
3. **Celda 69dd1aea (InMemoryVectorStore):**
   - ❌ `InMemoryVectorStore(OpenAIEmbeddings())`
   - ✅ `InMemoryVectorStore(embeddings)`
4. **Celda 964b9696 (LLM):**
   - ❌ `ChatOpenAI(model=llm_model, temperature=0.1)`
   - ✅ `ChatGroq(model=llm_model, temperature=0.1)`
5. **Celda 1779f900 (Chroma):**
   - ❌ `Chroma.from_documents(cleaned_texts, OpenAIEmbeddings())`
   - ✅ `Chroma.from_documents(cleaned_texts, embeddings)`
6. **Datos preparados:**
   - ✅ PDF copiado a `notebooks/data/Understanding_Climate_Change.pdf`
   - ✅ Dataset de películas en `../semantic-search/dataset.json` (ya existe)

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
- 🔄 **En progreso:** raglangchain.ipynb (RAG = búsqueda semántica + LLM)
  - ✅ Dependencias instaladas (pypdf, langsmith)
  - ✅ Notebook adaptado (OpenAI → Groq + HuggingFace)
  - ✅ PDF preparado en notebooks/data/
  - ⏳ Listo para ejecutar
- 🎯 **Objetivo:** Aprender 8 notebooks principales
- 📝 **Progreso:** 2/8 completado (25%), 3er notebook en progreso

---

**Última actualización:** 2025-11-06
**Sesión actual:** raglangchain.ipynb adaptado, listo para ejecutar
**Próxima sesión:** Ejecutar raglangchain.ipynb celda por celda
**Nota:** RAG combina búsqueda semántica + LLM para responder con contexto
