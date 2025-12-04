 # Objetivo

 Correr el script de entrenamiento de red neuronal desde líneas de comandos para luego integrarlo en una app.

 Para mayor referencia sobre los scripts comunicarse con su autora: Gabriela Perez <gperez@lifia.info.unlp.edu.ar>
 
 # Creemos un ambiente para aislar las depentedencias

Vamos a utilizar el modulo VirtualEnv

```
python -m venv .venv
source .venv/bin/activate
```


 # Instalar dependencias 

``pip3 install tensorflow scikit-learn matplotlib pandas ipython``

# Correr el script desde lineas de commandos

Primero convertimos el archivo de Notebook a Python

```
jupyter nbconvert --to python ClaseLeIA1.ipynb 
```

Corremos el script localmente:
```
python3 -m IPython ClaseLeIA1.py
```

Agregamos lineas para almacenar el modelo

````
import pickle
model_pkl_file = "model.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)

with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)
````

otra alternativa es utilizando tensorflow:

```
model.save("model.keras")
model = tf.keras.models.load_model("model.keras")
```

# Throbleshooting
En caso de este error:

`ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).`

Debemos convertir los arreglos a float32. Debemos cambiar de:


```
data = pd.get_dummies(data)
```
a:

```
data = pd.get_dummies(data)
data=data.astype(np.float32)
```

