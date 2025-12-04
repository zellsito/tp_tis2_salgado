FROM python:3.12
WORKDIR /code

# instalemos las dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copiemos el código fuente
COPY . /code/app

# creemos un usuario
RUN useradd app
USER app
#EXPOSE 5000



CMD ["fastapi", "run", "app/main.py", "--port", "80"]
#CMD ["ls"]