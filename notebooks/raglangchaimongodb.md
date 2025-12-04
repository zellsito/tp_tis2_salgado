# RAG con MongoDB Atlas - Documentación de Cambios

## 📋 Resumen

Notebook migrado de **MongoDB Local (Docker)** a **MongoDB Atlas (Cloud)** con índices vectoriales optimizados para búsquedas semánticas usando Atlas Vector Search.

## 🔄 Cambios Principales

### 1. Migración a MongoDB Atlas

**Antes (Local):**
```python
# MongoDB Local con Docker Compose
uri = f"mongodb://{user}:{password}@localhost:27017/?authSource=admin"
mongo_client = MongoClient(uri)
```

**Después (Atlas):**
```python
# MongoDB Atlas con Vector Search
uri = os.getenv("MONGO_URI", "mongodb+srv://user:...")
mongo_client = MongoClient(uri, server_api=ServerApi('1'))
```

### 2. Índices Vectoriales Habilitados

**Problema Original:**
- MongoDB local no soporta Atlas Vector Search
- Sin índices HNSW optimizados
- Búsquedas por fuerza bruta (lentas)

**Solución con Atlas:**
```python
# Crear índice vectorial HNSW (dimensión 384 para all-MiniLM-L6-v2)
try:
    mongo_vectorstore.create_vector_search_index(dimensions=384)
    print("✅ Índice vectorial creado exitosamente")
except Exception as e:
    if "already exists" in str(e).lower():
        print("ℹ️ Índice vectorial ya existe, continuando...")
```

**Beneficios:**
- ✅ Búsquedas vectoriales rápidas con HNSW
- ✅ Escalable a millones de documentos
- ✅ Búsquedas híbridas (semántica + filtros)
- ✅ Código idempotente (maneja índices existentes)

### 3. Re-ranking Compatible con LangChain

**Problema Original:**
```python
# ❌ TypeError: Expected a Runnable, callable or dict
class RerankedRetriever:  # No hereda de Runnable
    def invoke(self, query: str):
        ...
```

**Solución:**
```python
# ✅ Compatible con LCEL usando RunnableLambda
def create_reranked_retriever(base_retriever, top_n=3):
    def rerank_chain(query: str):
        docs = base_retriever.invoke(query)
        return rerank_documents(query, docs, top_n)

    return RunnableLambda(rerank_chain)

compression_retriever = create_reranked_retriever(my_retriever, top_n=3)
```

Ahora funciona correctamente en cadenas RAG:
```python
rag_chain1 = (
    {"context": compression_retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

### 4. Variable de Entorno

**Agregada en `.env`:**
```bash
# MongoDB Atlas (Cloud - con Vector Search)
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/?appName=Cluster0
```

## 🎯 Componentes del Stack

| Componente | Tecnología | Motivo |
|------------|-----------|--------|
| **Base de Datos** | MongoDB Atlas | Vector Search con HNSW |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Gratis, local, dim=384 |
| **LLM** | Groq (llama-3.1-8b-instant) | Gratis, rápido |
| **Re-ranking** | CrossEncoder (ms-marco-MiniLM-L-6-v2) | Mejora relevancia |
| **Framework** | LangChain 1.0+ | LCEL compatible |

## 📊 Arquitectura RAG

```
Query → Embeddings → MongoDB Atlas Vector Search (HNSW)
                          ↓
                     Top 5 Docs
                          ↓
                   CrossEncoder Re-ranking
                          ↓
                     Top 3 Docs
                          ↓
                    RAG Chain (Groq LLM)
                          ↓
                      Response
```

## 🚀 Ventajas de Atlas Vector Search

1. **Rendimiento:** Índices HNSW optimizados vs fuerza bruta
2. **Escalabilidad:** Millones de documentos sin degradación
3. **Búsquedas Híbridas:** Combina semántica + filtros de metadatos
4. **Persistencia:** Datos en la nube, accesibles desde cualquier lugar
5. **Gratis:** Tier M0 soporta hasta 512MB de datos

## 🔧 Configuración Necesaria

### 1. Instalar Dependencias
```bash
pip install pymongo langchain-mongodb sentence-transformers
```

### 2. Configurar `.env`
```bash
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/?appName=Cluster0
GROQ_API_KEY=your_groq_api_key
OPENAI_MODEL=llama-3.1-8b-instant
```

### 3. Crear Índice en Atlas (Opcional Manual)
Si prefieres crear el índice manualmente en la UI de Atlas:
1. Ve a tu cluster → Database → Browse Collections
2. Selecciona `langchain_test_db.langchain_test_vectorstores`
3. Crear Search Index → JSON Editor:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 384,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```

## 📝 Errores Corregidos

### Error 1: TypeError en RAG Chain
```
TypeError: Expected a Runnable, callable or dict.
Instead got an unsupported type: <class '__main__.RerankedRetriever'>
```

**Causa:** Clase personalizada no hereda de `Runnable`
**Solución:** Usar `RunnableLambda` para compatibilidad LCEL

### Error 2: Índice ya existe
```
Error al crear índice: Index already exists
```

**Causa:** Ejecutar `create_vector_search_index()` múltiples veces
**Solución:** Bloque `try-except` con detección de duplicados

## 🧪 Testing

**Búsqueda sin Re-ranking:**
```python
query = "What is the main cause of climate change?"
docs = my_retriever.invoke(query)  # Top 5
```

**Búsqueda con Re-ranking:**
```python
docs = compression_retriever.invoke(query)  # Top 3 re-rankeados
```

**RAG Completo:**
```python
result = rag_chain1.invoke(query)
```

## 📚 Recursos

- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/)
- [LangChain MongoDB Integration](https://python.langchain.com/docs/integrations/vectorstores/mongodb_atlas)
- [Sentence Transformers](https://www.sbert.net/)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)

## ✅ Checklist de Implementación

- [x] Migración a MongoDB Atlas
- [x] Configuración de Vector Search con HNSW
- [x] Índices vectoriales con manejo de duplicados
- [x] Re-ranking compatible con LCEL
- [x] HuggingFace embeddings (all-MiniLM-L6-v2)
- [x] Groq LLM integration
- [x] Variables de entorno configuradas
- [x] Documentación completa

---

**Última actualización:** 2025-11-08
**Versión LangChain:** 1.0+
**Python:** 3.11+
