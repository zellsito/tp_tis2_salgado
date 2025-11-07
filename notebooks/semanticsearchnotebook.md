# Aprendizaje: semanticsearchnotebook.ipynb

## 📋 Índice de Celdas

| # | ID | Tipo | Descripción | ¿Ejecutar? |
|---|-----|------|-------------|------------|
| 0 | 169f05af | Markdown | Título: "Semantic Search with ChromaDB" | No ejecutable |
| 1 | 482c51f4 | Code | Instalación de paquetes (comentada, ya instalado) | ✅ Ejecutar |
| 2 | b51057d3 | Markdown | "Import Required Libraries" | No ejecutable |
| 3 | 3a43fb47 | Code | Imports (chromadb, pandas, SentenceTransformer) | ✅ Ejecutar |
| 4 | cd5da066 | Markdown | "Setup Environment and API Keys" | No ejecutable |
| 5 | a3fbd451 | Code | Setup modelo local (all-MiniLM-L6-v2) | ✅ Ejecutar |
| 6 | 2ffba9e1 | Markdown | "Define File Paths" | No ejecutable |
| 7 | 4ced7289 | Code | Definir rutas (dataset.json, chroma_db/) | ✅ Ejecutar |
| 8 | 1df4275d | Markdown | "Load and Explore the Dataset" | No ejecutable |
| 9 | c2c5cc4b | Code | Cargar dataset.json en DataFrame | ✅ Ejecutar |
| 10 | 48fed881 | Markdown | "Preview the Data" | No ejecutable |
| 11 | 50ce16b3 | Code | Mostrar primeras 3 películas | ✅ Ejecutar |
| 12 | 321de5cf | Markdown | "Initialize ChromaDB" | No ejecutable |
| 13 | cd722937 | Code | Inicializar ChromaDB con SentenceTransformer | ✅ Ejecutar |
| 14 | 9dupv6wbi08 | Markdown | "Force Collection Rebuild (Optional)" | No ejecutable |
| 15 | ec4b4kl00d4 | Code | Eliminar colección existente (forzar recreación) | ✅ Ejecutar |
| 16 | 088d30c6 | Markdown | "Create or Load Collection" | No ejecutable |
| 17 | d3743ae9 | Code | Crear colección y generar embeddings | ✅ Ejecutar |
| 18 | 38de1556 | Markdown | "Define Search Function" | No ejecutable |
| 19 | 29e70d95 | Code | Definir funciones de búsqueda | ✅ Ejecutar |
| 20 | d6b18c8c | Markdown | "Test the Semantic Search" | No ejecutable |
| 21 | a7578575 | Code | Probar búsquedas semánticas | ✅ Ejecutar |
| 22 | 40409578 | Markdown | "Interactive Search" | No ejecutable |
| 23 | e3d17e7c | Code | Búsqueda interactiva personalizable | ✅ Ejecutar |
| 24 | 3726d9b2 | Markdown | "Advanced Search Analysis" | No ejecutable |
| 25 | da226254 | Code | Función análisis detallado con scores | ✅ Ejecutar |
| 26 | 713283a3 | Code | Ejecutar análisis detallado | ✅ Ejecutar |
| 27 | 0aafeb1f | Markdown | "Collection Statistics" | No ejecutable |
| 28 | e9be5ad3 | Code | Estadísticas de la colección | ✅ Ejecutar |
| 29 | 7e06cce9 | Markdown | "Cleanup and Summary" | No ejecutable |
| 30 | 5678ae8b | Code | Resumen final del sistema | ✅ Ejecutar |

**Total:** 31 celdas (16 ejecutables, 15 markdown)

---

## 🔧 Errores Encontrados y Corregidos

### Error 1: Embeddings de OpenAI (dimensión incompatible)
**Problema:** Dataset contenía embeddings pre-calculados de OpenAI (dimensión 1536), pero el notebook usa modelo local (dimensión 384)

```python
# ❌ Error al ejecutar búsqueda
InvalidArgumentError: Collection expecting embedding with dimension of 1536, got 384
```

**Causa:**
1. `dataset.json` original tenía embeddings de OpenAI pre-calculados
2. ChromaDB creó colección con dimensión 1536
3. Nuevo modelo local `all-MiniLM-L6-v2` genera dimensión 384
4. Dimensiones incompatibles

**Solución aplicada:**
1. **Limpiar dataset.json** (eliminar embeddings pre-calculados)
2. **Crear backup:** `dataset_original_with_openai_embeddings.json` (210KB)
3. **Nuevo dataset.json:** 8.6KB (sin embeddings)
4. **Eliminar colección ChromaDB** antigua
5. **Regenerar embeddings** con modelo local

---

### Error 2: Colección vacía (búsqueda sin resultados)
**Problema:** Búsqueda semántica no retornaba resultados

```python
# ❌ Búsqueda no encontraba nada
🔍 Search Results for: 'superhero adventure'
No results found.
```

**Causa:**
1. Colección existente estaba vacía (0 documentos)
2. Código entraba en bloque `try` (cargaba colección vacía)
3. NUNCA ejecutaba bloque `except` (que agrega documentos)

**Solución:**
Agregar celda de forzar recreación (celda 15) que elimina colección existente antes de crear nueva.

---

### Error 3: Uso de embeddings pre-calculados
**Problema:** Código intentaba usar `df.embedding.tolist()` que no existe en dataset limpio

```python
# ❌ Código antiguo (celda d3743ae9)
movies_collection.add(
    embeddings=df.embedding.tolist(),  # ❌ Campo no existe
    metadatas=metadatas
)

# ✅ Código actualizado
movies_collection.add(
    documents=documents,  # ChromaDB genera embeddings automáticamente
    metadatas=metadatas
)
```

---

## 📚 Conceptos Clave

### 1. **Embeddings (Vectores de texto)**
Representación numérica de texto en espacio vectorial de 384 dimensiones.
- Textos similares → vectores cercanos
- Permite búsqueda semántica (no solo keywords)
- Modelo usado: `all-MiniLM-L6-v2`

### 2. **ChromaDB (Base de datos vectorial)**
Base de datos especializada en almacenar y buscar embeddings.
- Almacenamiento persistente en disco
- Genera embeddings automáticamente
- Búsqueda por similitud vectorial

### 3. **Búsqueda Semántica (Similarity Search)**
Buscar por significado, no por palabras exactas.
- Distancia baja = más similar
- Usa distancia euclidiana
- Retorna top-k resultados más similares

### 4. **SentenceTransformers**
Modelo open-source de HuggingFace para generar embeddings.
- ✅ Gratuito (no requiere API key)
- ✅ Local (no envía datos externos)
- ✅ Rápido (~90MB modelo)

---

## 🎯 Resumen Ejecutivo

### Tecnologías Principales

| Tecnología | Propósito |
|------------|-----------|
| **ChromaDB** | Base de datos vectorial |
| **SentenceTransformers** | Generar embeddings |
| **all-MiniLM-L6-v2** | Modelo de embeddings (384 dim) |
| **Pandas** | Manipulación de datos |

### Métricas del Sistema

| Métrica | Valor |
|---------|-------|
| Total películas | 10 |
| Dimensión embeddings | 384 |
| Tamaño dataset | 8.6KB |
| Tiempo generación embeddings | ~5-10 segundos |

### Flujo de Datos

```
1. Cargar dataset.json
   ↓
2. Convertir a DataFrame
   ↓
3. Preparar textos (title + overview)
   ↓
4. ChromaDB genera embeddings
   ↓
5. Almacenar en colección
   ↓
6. Búsqueda: query → embedding → similarity
   ↓
7. Retornar resultados ordenados
```

---

## ✅ Checklist de Ejecución

- [x] Limpiar `dataset.json` (eliminar embeddings OpenAI)
- [x] Crear backup `dataset_original_with_openai_embeddings.json`
- [x] Ejecutar celdas 1-5 (setup e imports)
- [x] Ejecutar celda 7 (definir rutas)
- [x] Ejecutar celda 9 (cargar dataset)
- [x] Ejecutar celda 11 (preview datos)
- [x] Ejecutar celda 13 (inicializar ChromaDB)
- [x] Ejecutar celda 15 (eliminar colección antigua) ⚠️ CRÍTICO
- [x] Ejecutar celda 17 (crear colección y generar embeddings)
- [x] Ejecutar celda 19 (definir funciones)
- [x] Ejecutar celda 21 (probar búsquedas)
- [x] Ejecutar celda 23 (búsqueda interactiva)
- [x] Ejecutar celda 25-26 (análisis detallado)
- [x] Ejecutar celda 28 (estadísticas)
- [x] Ejecutar celda 30 (resumen)

---

## 🎓 Aprendizajes Clave

### 1. Embeddings vs Texto
Los embeddings son la representación matemática del significado del texto.

### 2. ChromaDB: Automatización
ChromaDB genera embeddings automáticamente si proporcionas `documents=`.

### 3. Persistencia
`PersistentClient` guarda en disco, no necesitas regenerar cada vez.

### 4. Búsqueda Semántica ≠ Keywords
Entiende el significado, no solo coincidencias literales.

### 5. Modelo Local vs API
Usar `all-MiniLM-L6-v2` es gratis y privado (vs OpenAI de pago).

### 6. Dimensionalidad importa
No puedes mezclar embeddings de diferentes dimensiones.

---

## 📝 Notas Finales

### Archivos Generados

```
notebooks/
├── dataset.json                                    # 8.6KB (limpio)
├── dataset_original_with_openai_embeddings.json    # 210KB (backup)
├── chroma_db/                                      # Base de datos vectorial
└── semanticsearchnotebook.ipynb                    # Notebook ejecutado
```

### Comparación con chatmodel.ipynb

| Aspecto | chatmodel.ipynb | semanticsearchnotebook.ipynb |
|---------|-----------------|------------------------------|
| **Enfoque** | LLMs y prompting | Búsqueda vectorial |
| **Tecnología** | LangChain + Groq | ChromaDB + SentenceTransformers |
| **Output** | Texto generado | Documentos similares |
| **Uso** | Chatbots, QA | Búsqueda, recomendación |

**Sinergia:** Estos notebooks se complementan para construir RAG:
1. semanticsearchnotebook: Buscar documentos relevantes
2. chatmodel: Generar respuestas basadas en esos documentos

---

**Próximo notebook recomendado:** `raglangchain.ipynb` (combina búsqueda semántica + LLM)
