import joblib
import numpy as np
import os
from src.agents.base_agent import Agent

class LightGBMAgent(Agent):
    name = "LightGBM Agent"
    color = Agent.MAGENTA

    def __init__(self):
        super().__init__(self.name, self.color)
        
        # Paths based on your screenshot structure
        base_path = "src/models/lightgbm_svd_3000"
        model_path = f"{base_path}/lightgbm_model.pkl"
        vec_path = f"{base_path}/tfidf_vectorizer.pkl"

        self.model = None
        self.vec = None

        try:
            self.log(f"Loading LightGBM from {base_path}...")
            # Load model and vectorizer
            self.model = joblib.load(model_path)
            self.vec = joblib.load(vec_path)
            self.log("LightGBM Model Loaded.")
        except Exception as e:
            self.log(f"Failed to load LightGBM: {e}")

    def _construct_training_prompt(self, data: dict) -> str:
        """
        Recreates the EXACT string format used during training.
        """
        # 1. System Instruction
        system_text = (
            "You are a helpful assistant that estimates the price of a product based on its title, "
            "category, features, description, and details. Return only the numeric price in USD, "
            "formatted with two decimals (e.g., 19.99)."
        )

        # 2. Fill in the fields
        user_text = (
            f"\n\nProduct Title: {data.get('title', '')}\n"
            f"Product Category: {data.get('category', 'General')}\n"
            f"Product Features: {data.get('features', '')}\n"
            f"Product Description: {data.get('description', '')}\n"
            f"Additional Details: {data.get('details', 'N/A')}\n\n"
            "Predict the price in USD (two decimals)."
        )
        
        return system_text + user_text

    def predict(self, deal_data: dict) -> float:
        """
        Predicts price based on the FULL structured prompt.
        """
        if not self.model or not self.vec:
            self.log("LightGBM Model not loaded, returning 0.0")
            return 0.0

        try:
            # 1. Reconstruct the Prompt String
            full_text_input = self._construct_training_prompt(deal_data)
            
            # 2. Vectorize the Full Text
            vector = self.vec.transform([full_text_input])
            
            # 3. Predict 
            price = self.model.predict(vector)[0]
            
            price = max(0.0, float(price))
            
            # 4. Round & Return
            final_price = round(price, 2)
            return final_price

        except Exception as e:
            self.log(f"LightGBM Error: {e}")
            return 0.0
