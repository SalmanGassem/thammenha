from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = FastAPI()

################ Loading Model ########################################

# # Load the model and scaler
# model = joblib.load('usa_model.joblib')
# scaler = joblib.load('scaler.joblib')

################ Model ################################################

# Define the data model for the input
class InputFeatures(BaseModel):
    brand: str
    model: str
    model_year: int
    milage: int
    fuel_type: str
    engine: str
    transmission: str
    ext_col: str
    int_col: str
    accident: str
    clean_title: str
    Vehicle_Age: int
    Mileage_per_Year: float
    Horsepower: float
    Engine_Size: float
    Power_to_Weight_Ratio: float
    Is_Luxury_Brand: int
    Accident_Impact: int

# Function to preprocess the input data
def preprocess_features(input_features: InputFeatures):
    dict_f = {
        'brand': input_features.brand,
        'model': input_features.model,
        'model_year': input_features.model_year,
        'milage': input_features.milage,
        'fuel_type': input_features.fuel_type,
        'engine': input_features.engine,
        'transmission': input_features.transmission,
        'ext_col': input_features.ext_col,
        'int_col': input_features.int_col,
        'accident': input_features.accident,
        'clean_title': input_features.clean_title,
        'Vehicle_Age': input_features.Vehicle_Age,
        'Mileage_per_Year': input_features.Mileage_per_Year,
        'Horsepower': input_features.Horsepower,
        'Engine_Size': input_features.Engine_Size,
        'Power_to_Weight_Ratio': input_features.Power_to_Weight_Ratio,
        'Is_Luxury_Brand': input_features.Is_Luxury_Brand,
        'Accident_Impact': input_features.Accident_Impact
    }

    df_processed = pd.DataFrame(dict_f)
    categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns
        
    for col in categorical_columns:
        df_processed[col] = scaler.fit_transform(df_processed[col].astype(str))
    
    return df_processed
    # Convert dictionary values to a list in the correct order
    # features_list = [dict_f[key] for key in sorted(dict_f)]

    # Scale the input features
    # scaled_features = scaler.transform([list(dict_f.values())])

    # return features_list

############### Predicting ############################################

# # KSA Prediction endpoint
# @app.post("/predict")
# async def predict(input_features: InputFeatures):
#     data = preprocess_features(input_features)
#     y_pred = model.predict(data)
#     return {"prediction": y_pred.tolist()[0]}

# # US Prediction endpoint
# @app.post("/predict")
# async def predict(input_features: InputFeatures):
#     data = preprocess_features(input_features)
#     y_pred = model.predict(data)
#     return {"prediction": y_pred.tolist()[0]}

# Fake Prediction endpoint
@app.post("/predict/ksa")
async def predict():
    return "87934"

# Fake Prediction endpoint
@app.post("/predict/us")
async def predict():
    return "72472"

#######################################################################

@app.get("/")
def root():
    return "Prediction API is working."

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
