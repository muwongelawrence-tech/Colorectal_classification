import bentoml

import numpy as np
from bentoml.io import Image
from bentoml.io import JSON
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

runner = bentoml.keras.get("keras_resnet50:latest").to_runner()

svc = bentoml.Service("keras_resnet50", runners=[runner])

@svc.api(input=Image(), output=JSON())
async def predict(img):
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = await runner.async_run(arr)
    return decode_predictions(preds, top=1)[0]