from typing import Union
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
import json as JSON
import numpy as np 

import tensorflow as tf

app = FastAPI()




@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


class Predict(BaseModel):
    sueldo_basico:int
    categoria:Literal["A","B","C"]
    ausencias:int
    cantidad_hijos:int


@app.put("/predict")
def predict(body:Predict):  
    array=[body.sueldo_basico,body.ausencias,body.cantidad_hijos]
    match body.categoria:
        case "A":
            array+= [1,0,0]
        case "B":
            array += [0, 1, 0]
        case "C":
            array+= [0,0,1]
    param = np.array([array]).astype(np.float32)

    model = tf.keras.models.load_model("../0_ML/model.keras")
    result=model.predict(param)
    res=JSON.dumps(result.tolist())
    return res


   
