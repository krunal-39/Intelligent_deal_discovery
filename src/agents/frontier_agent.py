import os
import re
import google.generativeai as genai
from src.agents.base_agent import Agent

class FrontierAgent(Agent):
    """
    Wraps Google Gemini 2.5 Flash-Lite.
    """
    name = "Frontier (Gemini)"
    color = Agent.BLUE

    def __init__(self):
        super().__init__(self.name, self.color)
        
        self.api_key = os.getenv("GEMINI_API_KEY_4")
        
        if not self.api_key:
            self.log("Error: GEMINI_API_KEY_4 not found in .env")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
            self.log("Gemini Model Initialized.")

    def predict(self, unified_prompt: str) -> float:
        """
        Sends the RAG-enriched prompt to Gemini.
        """
        if not self.api_key:
            return 0.0

        try:
            # 1. OPTIMIZATION: Force concise output via API config
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=10, 
                temperature=0.1      
            )
            
            response = self.model.generate_content(
                unified_prompt, 
                generation_config=generation_config
            )
            text = response.text.strip()
            
            # 2. IMPROVED PARSING STRATEGY
            # Matches: "$19.99", "$ 19.99"
            dollar_match = re.search(r'\$\s?(\d+(?:\.\d+)?)', text)
            if dollar_match:
                return round(float(dollar_match.group(1)), 2)

            # Priority B: Look for a number with a decimal point (Likely a price)
            # Matches: "19.99" inside "The price is 19.99"
            decimal_match = re.search(r'(\d+\.\d{1,2})', text)
            if decimal_match:
                return round(float(decimal_match.group(1)), 2)

            # Priority C: Fallback to the first number found
            nums = re.findall(r"\d+\.?\d*", text)
            if nums:
                return round(float(nums[0]), 2)
            
            return 0.0

        except Exception as e:
            self.log(f"Gemini Error: {e}")
            return 0.0
