# Setup del Proyecto - Jupyter Notebooks + Python + LangChain

## 📋 Resumen
Configuración completa del entorno para ejecutar `chatmodel.ipynb` usando:
- **Python 3.11.2** (entorno virtual)
- **Jupyter Notebooks** (en VS Code)
- **Groq API** (LLM gratuito)
- **HuggingFace Embeddings** (gratuito, local)

---

## 1. Verificar Python Instalado
```bash
python3 --version
# Output esperado: Python 3.11.2
```

---

## 2. Instalar Extensión Jupyter en VS Code
- Abrir VS Code
- Extensions (Ctrl+Shift+X)
- Buscar "Jupyter"
- Instalar extensión oficial de Microsoft

---

## 3. Crear Entorno Virtual
```bash
python3 -m venv .venv
```
**¿Qué hace?** Crea carpeta `.venv` con Python aislado para este proyecto.

---

## 4. Instalar Dependencias Base

### Jupyter
```bash
.venv/bin/pip install jupyter
```
**Qué instala:**
- `jupyter`: Framework completo para notebooks
- `ipykernel`: Motor que ejecuta código Python en las celdas

### LangChain Core
```bash
.venv/bin/pip install langchain langchain-core langchain-community
```
**Qué instala:**
- `langchain`: Framework principal para LLMs
- `langchain-core`: Núcleo de LangChain (prompts, parsers, etc.)
- `langchain-community`: Herramientas comunitarias

### Integraciones LLM
```bash
.venv/bin/pip install langchain-groq langchain-openai
```
**Qué instala:**
- `langchain-groq`: Integración con Groq (LLM gratuito)
- `langchain-openai`: Clases de OpenAI (usado en el notebook original)

### Base de Datos Vectorial
```bash
.venv/bin/pip install langchain-chroma
```
**Qué instala:**
- `langchain-chroma`: Base de datos vectorial para búsquedas semánticas

### Embeddings Locales (HuggingFace)
```bash
.venv/bin/pip install sentence-transformers langchain-huggingface
```
**Qué instala:**
- `sentence-transformers`: Modelos de embeddings de HuggingFace
- `langchain-huggingface`: Integración LangChain + HuggingFace
- **Incluye:** PyTorch, transformers, scikit-learn, scipy (~3GB total)

### Análisis de Datos
```bash
.venv/bin/pip install pandas
```
**Qué instala:**
- `pandas`: Análisis y manipulación de datos (DataFrames)

### Procesamiento de PDFs (para RAG)
```bash
.venv/bin/pip install pypdf
```
**Qué instala:**
- `pypdf`: Lector de archivos PDF para ingesta de documentos
- **Usado en:** raglangchain.ipynb (cargar PDFs como contexto)

### LangSmith (para prompts públicos y observabilidad)
**Nota:** `langsmith` ya se instaló en la sección de Utilidades.

**Uso adicional en RAG:**
- Descargar prompts públicos del hub con `client.pull_prompt()`
- Ejemplo: `hub_client.pull_prompt("rlm/rag-prompt")`
- **Usado en:** raglangchain.ipynb

### Utilidades
```bash
.venv/bin/pip install python-dotenv langsmith
```
**Qué instala:**
- `python-dotenv`: Carga variables desde archivo `.env`
- `langsmith`: Cliente de LangSmith (observabilidad, opcional)

---

## 5. Crear `.gitignore`
```bash
# Crear archivo .gitignore en la raíz del proyecto
```

**Contenido:**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Entorno virtual
.venv/
venv/
ENV/

# Jupyter
.ipynb_checkpoints/

# Variables de entorno
.env

# VS Code
.vscode/
```

---

## 6. Obtener Groq API Key

1. Ir a https://console.groq.com/
2. Crear cuenta (gratis, sin tarjeta)
3. Click "API Keys" → "Create API Key"
4. Copiar la key (empieza con `gsk_...`)

---

## 7. Crear Archivo `.env`

```bash
# Crear .env en la raíz del proyecto
```

**Contenido:**
```bash
# Groq API Key (obtenida en paso 6)
GROQ_API_KEY=gsk_tu_key_aqui

# Modelo a usar (actualizado a llama-3.1-8b-instant)
OPENAI_MODEL=llama-3.1-8b-instant
```

**Nota:** El `.gitignore` ya está configurado para NO subir este archivo a Git.

---

## 8. Modificar `chatmodel.ipynb`

### Cambio 1: Celda 5 (imports)
**Antes:**
```python
from langchain.memory import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings
```

**Después:**
```python
from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_openai import OpenAIEmbeddings  # ❌ OpenAI (de pago)
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ HuggingFace (gratis)
```

### Cambio 2: Celda 7 (configuración LLM)
**Antes:**
```python
llm = ChatOpenAI(model=llm_model, temperature=0.1)
# llm = ChatGroq(model=llm_model, temperature=0.1)
```

**Después:**
```python
# llm = ChatOpenAI(model=llm_model, temperature=0.1)
llm = ChatGroq(model=llm_model, temperature=0.1)
```

### Cambio 3: Celda 21 (embeddings)
**Antes:**
```python
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),  # ❌ Requiere OPENAI_API_KEY
    Chroma,
    k=1,
)
```

**Después:**
```python
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    # OpenAIEmbeddings(),  # ❌ Antiguo: requiere OPENAI_API_KEY
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),  # ✅ Nuevo: gratis
    Chroma,
    k=1,
)
```

---

## 9. Seleccionar Kernel en VS Code

1. Abrir `chatmodel.ipynb` en VS Code
2. Click "Select Kernel" (arriba derecha)
3. Elegir "Python Environments" → `.venv/bin/python`
4. Verificar que aparezcan botones ▶️ Play en cada celda

---

## 10. Ejecutar el Notebook

### Orden de ejecución:
1. **Reiniciar kernel** si estaba abierto (botón "Restart")
2. **Ejecutar celdas en orden:**
   - Celda 2: Imports básicos
   - Celda 3: `load_dotenv()` → debe mostrar `True`
   - Celda 4: SALTAR (comentada con `%pip install`)
   - Celda 5: Imports LangChain
   - Celda 7: Primera llamada a Groq (chiste)
   - Continuar en orden...

### Primera ejecución de Celda 21:
- Descargará modelo `all-MiniLM-L6-v2` (~90MB)
- Tardará unos segundos
- Ejecuciones siguientes usarán caché local

---

## 📦 Dependencias Instaladas (Resumen)

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `jupyter` | Latest | Framework notebooks |
| `langchain` | Latest | Framework LLM |
| `langchain-core` | Latest | Núcleo LangChain |
| `langchain-groq` | Latest | Integración Groq |
| `langchain-huggingface` | Latest | Integración HuggingFace |
| `langchain-chroma` | Latest | Base de datos vectorial |
| `sentence-transformers` | Latest | Embeddings locales |
| `pandas` | Latest | Análisis de datos |
| `python-dotenv` | Latest | Variables de entorno |

---

## 🔧 Troubleshooting

### Error: "No module named 'langchain_huggingface'"
**Solución:**
```bash
.venv/bin/pip install langchain-huggingface
```

### Error: "Model llama-3.1-70b-versatile has been decommissioned"
**Solución:** Actualizar `.env`
```bash
OPENAI_MODEL=llama-3.1-8b-instant
```

### Error: "OPENAI_API_KEY environment variable not set"
**Solución:** Usar `HuggingFaceEmbeddings` en vez de `OpenAIEmbeddings` (ya corregido en paso 8)

### Kernel no aparece en VS Code
**Solución:**
1. Cerrar VS Code completamente
2. Reabrir proyecto
3. Seleccionar kernel nuevamente

---

## ✅ Verificación Final

Lista de chequeo antes de ejecutar el notebook:

- [ ] Python 3.11.2 instalado
- [ ] Extensión Jupyter en VS Code
- [ ] Entorno virtual `.venv` creado
- [ ] Todas las dependencias instaladas
- [ ] Archivo `.env` creado con `GROQ_API_KEY` y `OPENAI_MODEL=llama-3.1-8b-instant`
- [ ] Archivo `.gitignore` creado
- [ ] `chatmodel.ipynb` modificado (3 cambios)
- [ ] Kernel `.venv/bin/python` seleccionado en VS Code

---

## 🎯 Próximos Pasos

1. Ejecutar notebook celda por celda
2. Revisar `notebooks/chatmodel.md` para entender cada celda
3. Revisar `notebooks/README.md` para guía de aprendizaje
4. Experimentar modificando prompts y parámetros
