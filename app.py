from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd 
import joblib
import numpy as np 


# KSA
def map_user_input_to_df(user_input):
        
    '''
    map the user input to the same features of the model

    '''
    mapping = {
        'Car_Brands': {
            'Lexus': 'Car_Brands_Brand:Lexus',
            'Mercedes': 'Car_Brands_Brand:Mercedes',
            'Range Rover': 'Car_Brands_Brand:Range Rover'
        },
        'Car_Models': {
            'LX': 'Car_Models_LX',
            'Land Cruiser': 'Car_Models_Land Cruiser',
            'Range Rover': 'Car_Models_Range Rover',
            'S Class': 'Car_Models_S Class'
        },
        'Car_Gear_Types': {
            'Automatic': 'Car_Gear_Types_Automatic',
            'CVT': 'Car_Gear_Types_CVT'
        },
        'Car_Drivetrains': {
            'AWD': 'Car_Drivetrains_AWD',
            'Double (4x4)': 'Car_Drivetrains_Double (4x4)',
            'FWD': 'Car_Drivetrains_FWD'
        },
        'Car_Extensions': {
            '500': 'Car_Extensions_500'
        },
        'Car_Exterior_Colors': {
            'Black': 'Car_Exterior_Colors_Black'
        },
        'Car_Interior_Colors': {
            'Camel': 'Car_Interior_Colors_Camel',
            'Grey': 'Car_Interior_Colors_Grey'
        },
        'Car_Origins': {
            'Saudi': 'Car_Origins_Saudi'
        }
    }

    result = pd.DataFrame({
        'Car_Kilometers': [user_input['Car_Kilometers']],
        'Car_Engine_Sizes': [user_input['Car_Engine_Sizes']],
        'Car_Seat_Numbers': [user_input['Car_Seat_Numbers']],
        'Car_Models_LX': [False],
        'Car_Models_Land Cruiser': [False],
        'Car_Models_Range Rover': [False],
        'Car_Models_S Class': [False],
        'Car_Brands_Brand:Lexus': [False],
        'Car_Brands_Brand:Mercedes': [False],
        'Car_Brands_Brand:Range Rover': [False],
        'Car_Gear_Types_Automatic': [False],
        'Car_Gear_Types_CVT': [False],
        'Car_Drivetrains_AWD': [False],
        'Car_Drivetrains_Double (4x4)': [False],
        'Car_Drivetrains_FWD': [False],
        'Car_Extensions_500': [False],
        'Car_Exterior_Colors_Black': [False],
        'Car_Interior_Colors_Camel': [False],
        'Car_Interior_Colors_Grey': [False],
        'Car_Origins_Saudi': [False]
    })
    
    for key, value in user_input.items():
        if key in mapping:
            if value in mapping[key]:
                result[mapping[key][value]] = True

    return result

# KSA
def scale(df):
    '''

    Scale numeric Data only!

    
    '''

    scaler = joblib.load('Scaler_SA.joblib')
    numeric_columns = ['Car_Kilometers', 'Car_Engine_Sizes', 'Car_Seat_Numbers']
    scaled_numeric = scaler.transform(df[numeric_columns])
    scaled_df = pd.DataFrame(scaled_numeric, columns=numeric_columns, index=df.index)
    df[numeric_columns] = scaled_df
    return df 

def predict_ksa(df): 
    from sklearn.metrics.pairwise import cosine_distances
    model = joblib.load('Knn_SA.joblib')
    scaled_data = pd.read_csv('Data.csv')
    if 'Unnamed: 0' in scaled_data.columns:
        scaled_data.drop(columns = 'Unnamed: 0', inplace = True)
    values = scaled_data.values
    distance = cosine_distances(df, values)

    return model.predict(distance)


 #metric='precomputed', n_neighbors=7)
app = FastAPI()

class CarInput(BaseModel):
    Car_Brands: str
    Car_Models: str
    Car_Years: int
    Car_Kilometers: int
    Car_Fuel_Types: str
    Car_Gear_Types: str
    Car_Engine_Sizes: float
    Car_Drivetrains: str
    Car_Extensions: str
    Car_Exterior_Colors: str
    Car_Interior_Colors: str
    Car_Seat_Numbers: int
    Car_Origins: str


@app.post("/predict/ksa")
async def map_car_data(car_input: CarInput):

    user_input = car_input.dict()
    
    df = map_user_input_to_df(user_input)
    df = scale(df)
    prediction = predict_ksa(df)
    print(prediction)
    return {'Predicted_Price': prediction[0]}

# Fake US Prediction endpoint
@app.post("/predict/us")
async def predict():
    return "72472"