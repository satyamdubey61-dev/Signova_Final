from core.engine import preprocess_and_predict, labels
from utils.logger import logger
import numpy as np

class PredictionService:
    @staticmethod
    def predict(img: np.ndarray):
        try:
            label, confidence = preprocess_and_predict(img)
            return label, confidence
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e
