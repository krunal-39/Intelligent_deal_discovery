import requests
import re
from src.agents.base_agent import Agent

class SpecialistAgent(Agent):
    """
    Wraps the Fine-Tuned LLaMA 3.1 model served via vLLM.
    """
    name = "Specialist (LLaMA)"
    color = Agent.RED
    
    # Point to your local vLLM instanc
    VLLM_URL = "http://localhost:8000/v1/completions"

    def __init__(self):
        super().__init__(self.name, self.color)

    def predict(self, unified_prompt: str) -> float:
        """
        Sends the RAG-enriched prompt to LLaMA.
        Returns price as float (e.g., 19.99).
        """
        payload = {
            "model": "specialist_llama",  
            "prompt": unified_prompt,
            "max_tokens": 10,       
            "temperature": 0.1 
        }
        
        try:
            res = requests.post(self.VLLM_URL, json=payload, timeout=20)
            
            if res.status_code == 200:
                text = res.json()['choices'][0]['text']
                
                # Extract first number found
                nums = re.findall(r"\d+\.?\d*", text)
                if nums:
                    price = float(nums[0])
                    return round(price, 2)
                
            self.log(f"vLLM Warning: Invalid response '{res.text}'")
            return 0.0
            
        except Exception as e:
            self.log(f"vLLM Connection Error: {e}")
            return 0.0
