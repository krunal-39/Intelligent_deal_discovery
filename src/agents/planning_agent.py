import os
import json
import time
from typing import List, Optional
from src.agents.base_agent import Agent
from src.agents.scanner_agent import ScannerAgent
from src.agents.ensemble_agent import EnsembleAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.messaging_agent import MessagingAgent

class PlanningAgent(Agent):
    name = "Planning Agent"
    color = Agent.GREEN
    MEMORY_FILE = "data/memory.jsonl"

    def __init__(self):
        super().__init__(self.name, self.color)
        
        self.log("Initializing Workforce...")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent() 
        self.evaluator = EvaluatorAgent()
        self.messenger = MessagingAgent()
        self.log("Planning Agent Ready.")

    def _load_memory(self) -> List[str]:
        if not os.path.exists(self.MEMORY_FILE):
            return []
        urls = []
        try:
            with open(self.MEMORY_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        urls.append(data.get("url"))
            return urls
        except Exception as e:
            self.log(f" Memory Load Error: {e}")
            return []

    def _save_to_memory(self, deal_data: dict):
        try:
            os.makedirs(os.path.dirname(self.MEMORY_FILE), exist_ok=True)
            with open(self.MEMORY_FILE, "a") as f:
                record = {
                    "url": deal_data['url'],
                    "title": deal_data['title'],
                    "verdict": deal_data['verdict'],
                    "timestamp": time.time()
                }
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.log(f" Memory Save Error: {e}")

    def process_deal(self, deal_data: dict) -> dict:
        """
        Estimates and Evaluates a deal.
        NOTE: Does NOT send the message anymore. Just returns the result.
        """
        self.log(f"--- Processing: {deal_data['title'][:40]}... ---")
        
        # 1. ENSEMBLE: Get AI Price
        prediction = self.ensemble.get_price(deal_data)
        ai_price = prediction['final_price']
        
        # 2. EVALUATOR: Judge the deal
        eval_result = self.evaluator.evaluate(
            current_price=deal_data['price'], 
            ai_estimated_price=ai_price
        )
        
        # 3. CONSOLIDATE
        final_result = {
            "title": deal_data['title'],
            "url": deal_data.get('url', 'N/A'),
            "current_price": deal_data['price'],
            "ai_fair_price": ai_price,
            "is_deal": eval_result['is_deal'],
            "verdict": eval_result['verdict'],
            "discount_amount": eval_result['discount_amount'],
            "discount_pct": eval_result['discount_pct'],
            "model_breakdown": prediction['breakdown']
        }
        
        # We still save to memory immediately so we don't process it again if script crashes
        if final_result['is_deal']:
            self._save_to_memory(final_result)
        else:
            self.log(f"Deal rejected ({eval_result['verdict']}).")
            
        return final_result

    def start_workflow(self, user_query: str = "best tech deals"):
        """
        1. Scans
        2. Processes
        3. Collects all deals
        4. Sends ONE summary message
        """
        self.log(f"Starting Workflow for Request: '{user_query}'")
        
        seen_urls = self._load_memory()
        found_deals = self.scanner.scan(memory=seen_urls, query=user_query)
        
        #  NEW LIST TO STORE VALID DEALS
        great_deals_list = []
        
        for deal in found_deals:
            deal_dict = {
                "title": deal.title,
                "price": deal.price,
                "url": deal.url,
                "description": deal.description,
                "category": deal.category,
                "features": deal.features,
                "details": deal.details
            }
            
            res = self.process_deal(deal_dict)
            
            #  COLLECT IF IT IS A DEAL
            if res['is_deal']:
                great_deals_list.append(res)
                
        # SEND ALL DEALS IN ONE BATCH AT THE END
        if len(great_deals_list) > 0:
            self.log(f"sending {len(great_deals_list)} deals to Telegram...")
            self.messenger.send_batch_summary(great_deals_list)
        else:
            self.log("No new deals found to send.")

        self.log("Workflow Cycle Complete.")
        return great_deals_list
