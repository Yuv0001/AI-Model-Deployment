from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Annotated
from tensorflow import keras
import numpy as np
from PIL import Image
from io import BytesIO


app = FastAPI()


DATABASE_URL = "mysql+pymysql://root:Yuv123@localhost:3306/classifiermodel"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(String(255))
    prediction = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

# Load the pre-trained model
model = keras.models.load_model("dog_cat_classifier.h5")
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



db_dependency = Annotated[Session,Depends(get_db)]

@app.post("/predict")
async def predict(file:UploadFile=File(...), db:db_dependency= Annotated[Session,Depends(get_db)]):
    try:
        image_data = await file.read()
        img = Image.open(BytesIO(image_data))
        img = img.resize((150, 150))  
        img_array = np.array(img) / 255.0  
        
        

        threshold = 0.5
        prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]
        prediction_label = 'DOG' if prediction > threshold else 'CAT'
        
        # print(prediction)

        
        log_entry = PredictionLog(input_data=f"Uploaded image: {file.filename}", prediction=prediction_label)
        db.add(log_entry)
        db.commit()

      
        return {"prediction": prediction_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
