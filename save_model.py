import bentoml
from tensorflow.keras.models import load_model

model_path = "vgg19_bs100_e10.h5"

model = load_model(model_path)

# colorectal_Cancer_Predictor = bentoml.keras.save_model("Colorectal_Cancer_Classifier", model)
colorectal_Cancer_Predictor = bentoml.tensorflow.save_model("Colorectal_Cancer_Classifier", model)

print(f"Model_Id:", colorectal_Cancer_Predictor)

