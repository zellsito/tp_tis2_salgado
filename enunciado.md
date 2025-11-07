Tópicos de Ingeniería de Software 2
2025

Enunciado
Este trabajo integrador consiste en el desarrollo en Python de una plataforma que ofrezca servicios Web (APIs) que permitan a profesionales médicos contar con asistencia para la interpretación de información clínica y la predicción de enfermedades de pacientes (que han sido dados de baja recientemente de un hospital). En particular se busca desarrollar un servicio Web que retorna una predicción sobre la probabilidad de que un paciente vaya a tener neumonía, en base a información de su historial de medicación y de notas médicas.

Para la implementación de este servicio se aplicarán técnicas de Machine Learning (clasificación) y de LLMs (Retrieval Augmented Generation)

El servicio solo deberá poder ser invocado por clientes de nuestra plataforma y por lo tanto deberán pasar una API Key.  Las invocaciones (request) HTTP deberán contar con el header HTTP 'Authorization' indicando la API key.  Si la API key no se encuentra en la invocación, ésta deberá ser rechazada.

Ejemplo de parámetros de entrada del servicio (aproximado):

POST /api/v1/predict/pneumonia
{
  "patient_id": "12345",
  "ethnicity": “white”,
  "gender": "male",
  "prescriptions": ["Amoxicillin", “Prednisone", etc.],
 "clinical_notes": "El paciente presenta tos persistente y dificultad respiratoria leve. Se observa presión arterial elevada." (opcional)
}

Nota: Las medicaciones pueden estar estandarizadas en base a un nomenclador (ej., GPI)

Ejemplo de retorno del servicio:
{
  "pneumonia": True,
  "score": 0.82,
 "explanation": "Según la presión elevada y antecedentes, el paciente presenta un riesgo alto de enfermedad ..."
}


Existen dos tipos de servicios que los médicos pueden contratar, que restringen la cantidad de solicitudes HTTP por minuto que el sistema está autorizado a resolver por minuto, y también la calidad de la respuesta:
-	FREEMIUM; 5 solicitudes por minuto (RPM), y el servicio simplemente retorna si/no (predicción de neumonía) y la consiguiente probabilidad del resultado
-	PREMIUM: 50 solicitudes por minuto (RPM), y el servicio puede incluir una explicación de porqué se generó ese resultado.


El servicio deberá satisfacer los siguientes requerimientos:
-	Deberá correr un modelo pre-entrenado (ya sea de ML, o un LLM), y permitir una actualización o ajuste periódico del modelo en base a nuevos datos o a feedback de un experto humano.
-	Todos las  invocaciones que reciba el servicio deberán ser controladas verificando dos aspectos:
-	Autorización. A partir de la API key, se verifica si existe registrada la API key en la base de datos del sistema.
-	Calidad de respuesta. De acuerdo a la suscripción del cliente tiene una limitación de invocaciones por segundo: FREEMIUM y PREMIUM.
-	Cada solicitud recibida deberá ser registrada en una bitácora (log). Capturando el tiempo que tomo para procesar el requerimiento HTTP de diagnóstico: iniciar el timer cuando se recibe la solicitud HTTP, procesar la autenticación de la key, correr la red neuronal, registrar el resultado en la bitácora, y retornar la respuesta.
-	Para datos de solo lectura y de poca volatilidad, se espera que se implemente una cache, de manera de minimizar el uso del servicio de inferencia (ante consultas repetitivas)
-	Para las notas médicas y la aplicación de RAG, se contará con una base de documentos (información no estructurada), tipo MongoDB o similar.
-	La solución deberá ser implementada en base a microservicios.


Entregables

La solución deberá contar con los siguientes entregables:
-	Instructivo para correr el proceso de entrenamiento del modelo.
-	Informe de diseño donde se presentan diagramas, aclaraciones sobre los requerimientos y toma de decisiones. 
-	Test HTTP para probar el funcionamiento de los servicios. Pueden utilizar HTTP, Postman, JMeter, cURL.

Entrega:
-	Implementar autenticación por API Key, con un único servicio (FREEMIUM).
-	Desarrollar (entrenar y servir) un modelo de ML de clasificación
-	Incluir el script para re-entregar el modelo de clasificación
-	Diseñar un esquema de caché para evitar búsquedas repetidas
-	Implementar el mecanismo de log
-	Implementar el servicio diferencial (FREEMIUM Y PREMIUM), con el limitador de invocaciones.
-	Desarrollar el servicio de LLM/RAG, incluyendo la base de documentos 
-	Adaptar el esquema de cache y logging al nuevo servicio
-	Documentación completa y casos de prueba.

Fechas de entrega
2026

