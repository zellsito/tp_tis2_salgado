# Plataforma base
FROM python:3.12-slim 
# Crea y cambia al directorio de la aplicación llamado /app
WORKDIR /app
# Instala Poetry
RUN pip install poetry
# Copia los archivos de dependencias al contenedor
COPY pyproject.toml poetry.lock*  ./
# Configura Poetry para no crear entornos virtuales
RUN poetry config virtualenvs.create false
# Instala las dependencias del proyecto
RUN poetry install  --no-root
#  Copia el código fuente de la aplicación al contenedor
COPY src/ ./src/
# Inicia la aplicación usando Uvicorn
CMD ["poetry", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5555", "--workers", "1"]