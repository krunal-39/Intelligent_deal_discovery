import xgboost as xgb
import pandas as pd
import numpy as np
from src.agents.base_agent import Agent
from src.agents.specialist_agent import SpecialistAgent
from src.agents.frontier_agent import FrontierAgent
from src.agents.lightgbm_agent import LightGBMAgent  
from src.agents.rag_utility import RAGUtility

class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self):
        super().__init__(self.name, self.color)
        
        # 1. Initialize the Tools & Workers
        self.rag = RAGUtility()
        self.specialist = SpecialistAgent()     # LLaMA
        self.frontier = FrontierAgent()         # Gemini
        self.lgbm = LightGBMAgent()             # LightGBM (The code you pasted belongs in lightgbm_agent.py)
        
        # 2. Load the Manager Brain (XGBoost)
        xgb_path = "src/models/ensemble/ensemble.json"
        
        try:
            self.xgb = xgb.XGBRegressor()
            self.xgb.load_model(xgb_path)
            self.log(f"✅ XGBoost Mixer Loaded from {xgb_path}.")
        except Exception as e:
            self.log(f"⚠️ XGBoost not found: {e}")
            self.log("   -> Will use simple average fallback.")
            self.xgb = None

    def get_price(self, deal_data: dict) -> dict:
        """
        Orchestrates the pricing process: RAG -> 3 Sub-Agents -> Aggregation.
        """
        title = deal_data.get('title', 'Unknown')
        self.log(f"Orchestrating prediction for: '{title}'")
        
        # --- Step A: Generate Unified Prompt ---
        unified_prompt = self.rag.get_unified_prompt(deal_data)
        
        # --- Step B: Parallel Inference ---
        
        # 1. Gemini (Frontier)
        gemini_price = self.frontier.predict(unified_prompt)
        
        # 2. LLaMA (Specialist)
        llama_price = self.specialist.predict(unified_prompt)
        
        # 3. LightGBM (Replacing Random Forest)
        # We pass the full deal_data dict because LightGBM needs title, description, etc.
        lgbm_price = self.lgbm.predict(deal_data)
        
        # --- Step C: Aggregate ---
        # Note: We map lgbm_price to the column previously used for 'rf_pred' 
        # so the pre-trained XGBoost model accepts the input.
        input_df = pd.DataFrame([[llama_price, gemini_price, lgbm_price]], 
                                columns=['llama_pred', 'gemini_pred', 'lgbm_pred'])
        
        final_price = 0.0
        
        if self.xgb:
            try:
                final_price = self.xgb.predict(input_df)[0] 
            except Exception as e:
                self.log(f"XGBoost Error: {e}, falling back to average")
                final_price = ((llama_price + gemini_price + lgbm_price) / 3 )
        else:
            final_price = (llama_price + gemini_price + lgbm_price) / 3
        
        final_price = round(float(final_price), 2)
        
        self.log(f"Consensus Price: ${final_price}")
        
        return {
            "final_price": final_price,
            "breakdown": {
                "llama": llama_price, 
                "gemini": gemini_price, 
                "lgbm": lgbm_price
            }
        }
