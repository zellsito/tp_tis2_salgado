# Proyecto TIS2 - Salgado

<!-- TODO: Completar descripción del proyecto real -->

---

## 📋 Estado del Proyecto

🚧 **En desarrollo** - Proyecto principal pendiente de implementación.

---

## 📂 Estructura del Repositorio

```
tp_tis2_salgado/
├── notebooks/               # 📚 Aprendizaje de Jupyter Notebooks + LangChain
│   ├── chatmodel.ipynb      # ✅ Completado
│   ├── chatmodel.md         # ✅ Documentación completa
│   └── README.md            # Guía de aprendizaje
├── setup.md                 # Configuración del entorno Python
├── PROMPT_CONTEXTO.md       # 🔄 Prompt para retomar desde 0
├── .env                     # Variables de entorno (no en Git)
├── .gitignore               # Archivos ignorados
└── README.md                # Este archivo
```

---

## 🎓 Aprendizaje Previo

Antes de comenzar el proyecto principal, se realizó un aprendizaje de:
- Jupyter Notebooks
- Python + LangChain
- Groq API (LLM gratuito)
- HuggingFace Embeddings

**Ver carpeta `notebooks/` para toda la documentación del aprendizaje.**

---

## 🚀 Configuración del Entorno

### Requisitos
- Python 3.11+
- VS Code con extensión Jupyter
- Cuenta Groq (gratuita)

### Setup Rápido
```bash
# 1. Crear entorno virtual
python3 -m venv .venv

# 2. Instalar dependencias (ver setup.md para lista completa)
.venv/bin/pip install jupyter langchain langchain-groq python-dotenv

# 3. Configurar .env
# GROQ_API_KEY=tu_key_aqui
# OPENAI_MODEL=llama-3.1-8b-instant
```

**Para setup completo:** Ver `setup.md`

---

## 📚 Documentación

| Archivo | Descripción |
|---------|-------------|
| `setup.md` | Configuración completa del entorno |
| `PROMPT_CONTEXTO.md` | 🔄 Prompt para retomar trabajo (contexto completo) |
| `notebooks/README.md` | Guía de aprendizaje de Jupyter + LangChain |
| `notebooks/chatmodel.md` | Explicación detallada de `chatmodel.ipynb` |

---

## 🛠️ Tecnologías (hasta ahora)

- Python 3.11.2
- Jupyter Notebooks
- LangChain
- Groq (LLM)
- HuggingFace (Embeddings)

---

## 📝 TODO

- [ ] Definir alcance del proyecto principal
- [ ] Diseñar arquitectura
- [ ] Implementar funcionalidades core
- [ ] Documentar proyecto principal

---

## 🎯 Próximos Pasos

1. Completar aprendizaje de notebooks (ver `notebooks/README.md`)
2. Definir requisitos del proyecto TIS2
3. Comenzar implementación

---

**Nota:** La carpeta `notebooks/` contiene material de aprendizaje, no es parte del proyecto final.
