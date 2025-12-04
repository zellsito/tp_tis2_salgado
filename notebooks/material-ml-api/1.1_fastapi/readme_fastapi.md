# Fuentes

Principalmente las de FastAPI

# crear el proyecto y configurar el env
Referencia: https://fastapi.tiangolo.com/
```
mkdir app
cd app
python3 -m venv .venv
```

Activar

``source .venv/bin/activate``

Instalar Flask

``pip install "fastapi[standard]"``

# Empezamos a programar a app

Creamos la carpeta de código fuente

``mkdir flaskr``

Application factory:`flaskr/__init__.py`


```
from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
```



Corramos la app

`fastapi dev main.py`

La probamos

`http://127.0.0.1:8000/docs`

# Endpoint predict

Definimos un endpoint para las opciones GET/POST

```

import json as JSON
@app.put("/predict")
def predict():  
   
    return JSON.dumps("value")

```

Instalamos algunas librerias necesarias dentro del proyecto
```
pip3 install tensorflow pandas
```

Suponiendo que el servicio va a ser invocado con la siguiente URL, extraemos los parámetros de la URL:

`http://127.0.0.1:5000/predict?sueldo_basico=34000&categoria=B&ausencias=0&cantidad_hijos=2&`


from typing import Literal
from pydantic import BaseModel

```
class Predict(BaseModel):
    sueldo_basico:float
    categoria:Literal["A","B","C"]
    ausencias:int
    cantidad_hijos:int

```



Una vez que contamos con todos los parámetros, nos disponemos a producir un array Numpy para que pueda ser utilizado con la fonción `model.predict()``

```
        import numpy as np #Manejar los arreglos con los datos
        
        array=[sueldo_basico,ausencias,cantidad_hijos]
```

Dado que la categoria es un flag, tenemos que armar la cola del vector:
```
        match categoria:
                case "A":
                    array+= [1,0,0]
                case "B":
                    array += [0, 1, 0]
                case "C":
                    array+= [0,0,1]
```

Finalmente preparamos los parametros para predecir:
```
        param = np.array([array]).astype(np.float32)

```



Finalmente ajustamos el retorno:

```
return JSON.dumps(param.tolist())

```

# Cargamos el modelo al inicializar la app

Una vez entrenado el modelo, podríamos utilizarlo en nuestro servicio. Para ello, vamos a leer el modelo desde un archivo y dejarlo disponible en una variable local.

```
   import tensorflow as tf

    model = tf.keras.models.load_model("../0_ML/model.keras")
```

Luego vamos a ajustar la funcion `predict()` para utilizar el modelo y luego retornar el resultado.

```
        result=model.predict(param)
        return JSON.dumps(result.tolist())

```


# Incorporamos una db

En primer lugar, debemos iniciar una base de datos. Existen dos opciones:

## Utilizar una instancia local de MongoDB

Se podría utilizar docker e iniciar un contenedor con el siguiente comando:

```
docker run -d --network some-network --name some-mongo \
	-e MONGO_INITDB_ROOT_USERNAME=mongoadmin \
	-e MONGO_INITDB_ROOT_PASSWORD=secret \
	mongo`
```

## Utilzar una instancia en la nube

Una opción en la nube es https://cloud.mongodb.com/ que brinda un servicio de Mongo DB.
Los pasos para activar una cuenta son:

1. Registrar una cuenta
2. Obtener la URL de conexión accediendo a
    1. Elegir `Database` desde el panel izquierdo
    2. Presionar `Connect``
    3. Elegir `Driver` y capturar el string de conexión de la sección 3.
3. Ajustar las credenciales (usuario/clave).
    1. Elegir `Database Access` desde el panel isquierdo.
    2. Presionar `Edit` en el usuario existente
    3. Ajustemos el password.

## Integración en nuestra aplicación

Instalemos la librería de Mongo: `pip3 install pymongo`


Definamos una función de logging:

```
import os
from pymongo import MongoClient, MongoClient
client = MongoClient(os.environ["MONGODB_URL"])
db = client.test_database

logs_collection = db.get_collection("logs")
def log(msg):
    logs_collection.insert_one(msg)

```

Exportamos la URL de conexión para que funcione el código anterior:

```
export MONGODB_URL="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
```

Posteriormente, registramos los parámetros y la salida del llamado al servicio en la base de datos Mongo.

```
        from datetime 


       
        log( 
            {
                "timestamp": datetime.now().isoformat(), # formateo de la fecha
                "params": param[0].tolist(),  #capturamos los parámetros de la invocación
                "response": result[0].tolist(), #capturamos el resultado
            }
        )
```

Por ultimo podemos exponer un servicio para conocer todas las invocaciones registradas en el sistema. Para ello vamos a exponer el siguiente endpoint:

`http://127.0.0.1:5000/requests`

El nuevo endpoint podría ser:
```
@app.get("/logs")
def get_logs():
    logs = list(logs_collection.find({},{"_id":0}))
    return logs    

```