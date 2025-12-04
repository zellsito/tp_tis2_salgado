# Aprendizaje: Jupyter Notebooks + LangChain + Groq

Documentación completa del aprendizaje de Jupyter Notebooks ejecutando paso a paso.

---

## 📁 Archivos en esta Carpeta

```
notebooks/
├── chatmodel.ipynb                  # ✅ Notebook prompting + LangChain (COMPLETADO)
├── chatmodel.md                     # ✅ Documentación completa
├── semanticsearchnotebook.ipynb     # ✅ Notebook búsqueda semántica (COMPLETADO)
├── semanticsearchnotebook.md        # ✅ Documentación completa
├── raglangchain.ipynb               # ✅ Notebook RAG + re-ranking (COMPLETADO)
├── raglangchain.md                  # ✅ Documentación completa
├── dataset.json                     # Dataset limpio (8.6KB, sin embeddings)
├── dataset_original_with_openai_embeddings.json  # Backup (210KB)
├── data/
│   └── Understanding_Climate_Change.pdf  # PDF para RAG (206KB)
├── chroma_db/                       # Base de datos vectorial ChromaDB
└── README.md                        # Este archivo
```

---

## 🚀 Inicio Rápido

### 1. Configurar Entorno
Sigue **todos los pasos** de `../setup.md` para:
- Crear entorno virtual
- Instalar dependencias
- Configurar Groq API key
- Modificar notebook

### 2. Ejecutar Notebook
1. Abrir `chatmodel.ipynb` en VS Code
2. Seleccionar kernel `.venv/bin/python`
3. Ejecutar celdas en orden (2, 3, 5, 7, 9, 11...)

### 3. Estudiar Conceptos
Revisar `chatmodel.md` para entender:
- Qué hace cada celda
- Conceptos de LangChain (chaineo, prompts, memoria)
- Comparaciones con TypeScript

---

## 📚 Documentación

| Archivo | Propósito |
|---------|-----------|
| **../setup.md** | Configuración paso a paso del entorno (Python, dependencias, API keys) |
| **chatmodel.md** | Explicación línea por línea del notebook + conceptos clave |

---

## 🛠️ Tecnologías

- **Python 3.11.2** - Lenguaje de programación
- **Jupyter Notebooks** - Entorno interactivo
- **LangChain** - Framework para LLMs
- **Groq** - LLM gratuito (llama-3.1-8b-instant)
- **HuggingFace** - Embeddings locales gratuitos

---

## ✅ Requisitos

- Python 3.11+
- VS Code con extensión Jupyter
- Cuenta en Groq (gratuita)
- ~3GB espacio libre (dependencias)

---

## 📖 Notebooks Completados

### 1. chatmodel.ipynb ✅
**Temas:** Prompting, chains, memoria, few-shot, structured output

| # | Descripción | Concepto |
|---|-------------|----------|
| 2 | Imports básicos | Setup |
| 3 | `load_dotenv()` | Variables de entorno |
| 5 | Imports LangChain | Framework |
| 7 | Primera llamada a Groq | LLM básico |
| 9 | Traducción con historial | Chaineo + Memoria |
| 11 | Zero-shot QA | Prompting sin ejemplos |
| 14 | Contextualización | Memoria + Reescritura |
| 21 | Selector semántico | Embeddings + Few-shot |
| 25 | Structured output | Pydantic |

**Documentación:** `chatmodel.md`

---

### 2. semanticsearchnotebook.ipynb ✅
**Temas:** Búsqueda semántica, embeddings, ChromaDB, vectorización

| # | Descripción | Concepto |
|---|-------------|----------|
| 3 | Imports | chromadb, pandas, SentenceTransformer |
| 5 | Setup modelo | all-MiniLM-L6-v2 (local, gratis) |
| 9 | Cargar dataset | JSON → DataFrame |
| 13 | Init ChromaDB | Base de datos vectorial |
| 15 | Forzar recreación | Eliminar colección antigua |
| 17 | Crear colección | Generar embeddings automáticamente |
| 21 | Probar búsquedas | "superhero adventure", "horror movie" |
| 23 | Búsqueda interactiva | Personalizable |
| 26 | Análisis detallado | Scores de similitud |

**Documentación:** `semanticsearchnotebook.md`

---

### 3. raglangchain.ipynb ✅
**Temas:** RAG (Retrieval Augmented Generation), re-ranking, visualización

| # | Descripción | Concepto |
|---|-------------|----------|
| Imports | Setup completo | LangChain 1.0+, Groq, HuggingFace |
| Movies | Búsqueda semántica básica | InMemoryVectorStore + metadata |
| RAG Chain | LLM + Retriever | {"context": retriever, "question": input} |
| PDF Loading | Cargar y procesar PDF | PyPDFLoader + chunking (97 chunks) |
| ChromaDB | Vector store persistente | Embeddings locales |
| Re-ranking | CrossEncoder manual | ms-marco-MiniLM-L-6-v2 |
| RAGxplorer | Visualización 2D | UMAP + plotly |

**Documentación:** `raglangchain.md`

---

## 🎓 Aprendizajes Clave

### Sobre Jupyter Notebooks
- Los notebooks combinan código, texto y visualizaciones
- El kernel se reinicia al cerrar → re-ejecutar celdas al abrir
- Ejecutar siempre de arriba hacia abajo
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

## 🔧 Troubleshooting

### Error: Module not found
Revisar `../setup.md` sección "Troubleshooting"

### Error: API key
Verificar que `.env` tenga `GROQ_API_KEY` válida

### Kernel no aparece
Reiniciar VS Code y reseleccionar kernel

---

## 🎯 Próximos Pasos

1. ✅ Completar ejecución del notebook
2. Experimentar modificando prompts
3. Probar diferentes modelos de Groq
4. Crear tus propios ejemplos few-shot
5. Explorar RAG (Retrieval-Augmented Generation)

---

## 📝 Recursos

### Documentación Oficial
- **LangChain:** https://python.langchain.com/docs/
- **Groq:** https://console.groq.com/docs/
- **HuggingFace:** https://huggingface.co/docs
- **Jupyter:** https://jupyter.org/documentation

### Modelos Recomendados
- **Groq (chat):** `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`
- **HuggingFace (embeddings):** `all-MiniLM-L6-v2`, `all-mpnet-base-v2`

---

## ✅ Progreso de Aprendizaje

### Notebooks Completados
- [x] **chatmodel.ipynb** - Prompting y LangChain básico
- [x] **semanticsearchnotebook.ipynb** - Búsqueda semántica con ChromaDB
- [x] **raglangchain.ipynb** - RAG + Re-ranking + Visualización
- [x] **react-web-search.ipynb** - ReAct Agents + Web Search con Tavily
- [x] **agentic-rag.ipynb** - Agentic RAG con flujo de decisiones inteligentes

---

## 📊 Inventario Completo de Notebooks

### Notebooks de LLM/LangChain (Prioridad Alta)
| # | Notebook | Tamaño | Estado | Complejidad |
|---|----------|--------|--------|-------------|
| 1 | chatmodel.ipynb | 13KB | ✅ COMPLETADO | Baja |
| 2 | semanticsearchnotebook.ipynb | 32KB | ✅ COMPLETADO | Baja-Media |
| 3 | raglangchain.ipynb | 33KB | ✅ COMPLETADO | Media |
| 4 | react-web-search.ipynb | 49KB | ✅ COMPLETADO | Media |
| 5 | agentic-rag.ipynb | 393KB | ✅ COMPLETADO | Alta |
| 6 | raglangchaimongodb.ipynb | 263KB | 📌 RECOMENDADO | Media-Alta |
| 7 | sql-agent.ipynb | 1.1MB | ⏳ PENDIENTE | Alta |
| 8 | langchainmultiagentcollaboration.ipynb | 1.1MB | ⏳ PENDIENTE | Muy Alta |

### Notebooks de ML/Data Science (Prioridad Baja)
| # | Notebook | Tamaño | Estado | Tema |
|---|----------|--------|--------|------|
| 9 | pneumoniapreprocessing.ipynb | 47KB | ⏸️ OPCIONAL | Preprocesamiento imágenes |
| 10 | salarypredictionregression.ipynb | 144KB | ⏸️ OPCIONAL | Regresión |
| 11 | customerchurnclassification-fs.ipynb | 529KB | ⏸️ OPCIONAL | Clasificación |

**Total:** 11 notebooks (5 completados, 6 pendientes)

---

## 🎯 Próximo Notebook Recomendado

### Opción A: raglangchaimongodb.ipynb (RECOMENDADA)
**Por qué:**
- ✅ Siguiente paso lógico después de RAG básico y agentic-rag
- ✅ Introduce persistencia con MongoDB (concepto importante)
- ✅ Tamaño mediano (263KB) - no tan intimidante como sql-agent (1.1MB)
- ✅ Complejidad Media-Alta - desafiante pero abordable
- ✅ Combina RAG + Base de datos (útil para proyectos reales)

**Temas que cubre:**
- RAG con vector store en MongoDB
- Persistencia de embeddings
- Queries combinando búsqueda vectorial + filtros tradicionales
- Integración LangChain + MongoDB Atlas

### Opción B: sql-agent.ipynb
**Consideraciones:**
- ⚠️ 1.1MB de notebook (muy extenso)
- ⚠️ Complejidad Alta
- ✅ Prerequisitos cumplidos (react-web-search + agentic-rag)
- ✅ Tema importante: Agentes + SQL + LangGraph

**Recomendación:** Dejar para después de raglangchaimongodb

---

## 📝 Orden de Aprendizaje Sugerido (Actualizado)

### Fase 1: Fundamentos ✅ COMPLETADA
1. ✅ chatmodel.ipynb - Prompting básico, chains, memoria
2. ✅ semanticsearchnotebook.ipynb - Embeddings, búsqueda semántica
3. ✅ raglangchain.ipynb - RAG + re-ranking

### Fase 2: Agentes ✅ COMPLETADA
4. ✅ react-web-search.ipynb - ReAct agents + Tavily
5. ✅ agentic-rag.ipynb - RAG con flujo de decisiones inteligentes

### Fase 3: Persistencia & Integración 🎯 EN PROGRESO
6. 📌 **raglangchaimongodb.ipynb** (SIGUIENTE RECOMENDADO)
7. ⏳ sql-agent.ipynb - Agentes + SQL + LangGraph

### Fase 4: Multi-Agentes Avanzado
8. ⏳ langchainmultiagentcollaboration.ipynb - Colaboración entre agentes

### Fase 5: ML/Data Science (Opcional)
9. ⏸️ pneumoniapreprocessing.ipynb
10. ⏸️ salarypredictionregression.ipynb
11. ⏸️ customerchurnclassification-fs.ipynb

**Progreso LLM/LangChain:** 5/8 completados (62.5%)
**Progreso Total:** 5/11 notebooks (45.5%)

---

**Nota:** Este es un ejercicio de aprendizaje. Para el proyecto real del TP, ver `../README.md`.
