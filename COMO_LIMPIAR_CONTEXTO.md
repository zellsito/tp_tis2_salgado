# Cómo Limpiar el Contexto de Claude Code

## ❓ ¿Por qué limpiar el contexto?

- **Ahorro de tokens:** Claude Code tiene un límite de 200,000 tokens por conversación
- **Conversaciones largas:** Acumulan mucho contexto innecesario
- **Mejor rendimiento:** Conversaciones más ágiles
- **Tokens usados actualmente:** ~103,000 / 200,000 (51%)

---

## 🔄 Cuándo Limpiar el Contexto

### Limpiar cuando:
- ✅ Has completado una tarea grande (ej: un notebook completo)
- ✅ Los tokens superan 150,000 (~75%)
- ✅ Vas a cambiar de tema/notebook
- ✅ Sientes que las respuestas son más lentas

### NO limpiar cuando:
- ❌ Estás en medio de una tarea
- ❌ Hay errores sin resolver
- ❌ Tokens < 100,000 (~50%)

---

## 📋 Método 1: Reiniciar Sesión (Recomendado)

### Pasos:

1. **Guardar trabajo actual**
   - Todo ya está documentado en archivos `.md`
   - Verificar que `PROMPT_CONTEXTO.md` existe

2. **Cerrar Claude Code**
   - En VS Code: Cerrar el panel de Claude Code
   - O reiniciar VS Code completamente

3. **Abrir nueva sesión**
   - Abrir panel de Claude Code
   - Pegar contenido de `PROMPT_CONTEXTO.md`
   - Claude Code cargará todo el contexto desde el prompt

### Ventajas:
- ✅ Contexto limpio (0 tokens usados)
- ✅ Toda la información en el prompt
- ✅ Sin perder progreso

### Desventajas:
- ❌ Tienes que copiar/pegar el prompt

---

## 📋 Método 2: Comando /clear (Si está disponible)

Algunos sistemas tienen comando `/clear` para limpiar historial.

**Verificar:**
```
/help
```

Si aparece `/clear`, úsalo. Sino, usa Método 1.

---

## 📋 Método 3: Nueva Ventana de VS Code

1. Guardar todo
2. Cerrar VS Code
3. Abrir nuevo VS Code
4. Abrir proyecto
5. Pegar `PROMPT_CONTEXTO.md` en Claude Code

---

## 🎯 Flujo Recomendado para Este Proyecto

### Cada vez que completes un notebook:

1. **Verificar que todo esté documentado**
   - `notebooks/<nombre>.md` creado
   - `notebooks/README.md` actualizado
   - No hay errores pendientes

2. **Actualizar PROMPT_CONTEXTO.md**
   - Marcar notebook como ✅ completado
   - Actualizar "Próximo paso"
   - Agregar aprendizajes clave si es necesario

3. **Revisar tokens usados**
   - Si > 150,000 tokens → limpiar contexto
   - Si < 150,000 tokens → continuar

4. **Limpiar contexto (si es necesario)**
   - Cerrar Claude Code
   - Reabrir y pegar `PROMPT_CONTEXTO.md`
   - Verificar que cargó correctamente

---

## ✅ Checklist Antes de Limpiar

- [ ] Todo el trabajo está en archivos `.md` (no solo en el chat)
- [ ] `PROMPT_CONTEXTO.md` está actualizado
- [ ] No hay errores sin resolver
- [ ] No estás en medio de una tarea
- [ ] Tokens > 150,000 (~75%)

---

## 🔄 Prompt para Retomar

**Archivo:** `PROMPT_CONTEXTO.md`

**Contiene:**
- Objetivo del proyecto
- Metodología de trabajo
- Estructura del proyecto
- Notebooks completados
- Siguiente notebook recomendado
- Orden de aprendizaje
- Configuración actual
- Errores corregidos
- Estado actual

**Uso:**
1. Copiar contenido completo de `PROMPT_CONTEXTO.md`
2. Pegar en nueva sesión de Claude Code
3. Claude Code entenderá todo el contexto

---

## 📊 Estado Actual

- **Tokens usados:** ~103,000 / 200,000 (51%)
- **Notebook completado:** chatmodel.ipynb ✅
- **Siguiente:** semanticsearchnotebook.ipynb
- **Acción recomendada:** Continuar sin limpiar (tenés ~97k tokens libres)

---

## 💡 Tips

- **Documenta todo:** Así no dependes del contexto de Claude Code
- **Usa PROMPT_CONTEXTO.md:** Mantenlo actualizado siempre
- **Sé proactivo:** Limpia contexto cuando termines una tarea grande
- **No tengas miedo:** Con el prompt guardado, no perdés nada

---

## 🎯 Próximo Paso

**NO limpiar contexto ahora.** Tenés suficientes tokens.

**Seguir con:** `semanticsearchnotebook.ipynb`

**Limpiar contexto después de:** Completar 2-3 notebooks más (~200k tokens).
