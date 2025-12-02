import os
import json
import google.generativeai as genai
from typing import List
from src.agents.base_agent import Agent
from src.agents.deals import ScrapedDeal, Deal

class ScannerAgent(Agent):
    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        super().__init__(name="Scanner Agent", color=self.CYAN)
        
        # Use Gemini Key 4 
        api_key = os.getenv("GEMINI_API_KEY_4")
        if not api_key:
            self.log("Warning: GEMINI_API_KEY_4 not found.")
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("models/gemini-2.5-flash")
            
        self.log("Scanner Agent is ready.")

    def _filter_deals_with_llm(self, scraped_deals: List[ScrapedDeal], query: str) -> List[Deal]:
        """
        Uses Gemini to select deals based on the USER QUERY.
        """
        if not scraped_deals:
            return []

        # Pass FULL description so Gemini can see specs
        
        deal_text = "\n\n".join([
            f"ID: {i}\nTitle: {d.title}\nPrice: ${d.price}\nStore: {d.retailer}\nDesc: {d.description}"
            for i, d in enumerate(scraped_deals)
            if d.price > 0 
        ])

        #  DYNAMIC PROMPT INJECTED HERE
        prompt = f"""
        You are a Deal Hunter. 
        I am looking for deals matching this specific request: "{query}"
        
        CRITICAL FILTERING RULES:
        1. Select ONLY specific physical products (e.g., "Dell Inspiron 15", "Sony Headphones").
        2. DO NOT select General Sales Events, Bundles, or Lists (e.g., "Cyber Monday Deals", "Up to 50% off", "Flash Sale").
        3. DO NOT select items where the title implies a collection (e.g., "Laptops starting at $200").
        4. Select up to 15 of the BEST matches. If fewer than 15 match, return only the good ones.

        From the list below, select the matches.
        
        For each selected deal, extract/infer the following fields based on the description:
        1. Clean Title
        2. Category (e.g., Computers, Electronics, Accessories)
        3. Features (Bullet points of technical specs: RAM, CPU, Material, etc.)
        4. Description (A flowing summary paragraph)
        5. Details (Tech specs like Brand, Color, Dimensions, Weight if available, otherwise 'N/A')
        
        Return a JSON list strictly in this format:
        [
            {{
                "original_id": 0, 
                "title": "Clean Title",
                "category": "Category Name",
                "features": "Feature list text",
                "description": "Summary text",
                "details": "Tech specs text"
            }}
        ]

        RAW DEALS:
        {deal_text}
        """

        try:
            self.log(f"Filtering deals for query: '{query}'...")
            # Increase output tokens to handle 10 items
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json",
                max_output_tokens=16384
            )
            
            response = self.model.generate_content(prompt, generation_config=generation_config)
            # Cleanup JSON markdown if present
            text = response.text.replace("```json", "").replace("```", "").strip()
            
            # Handle empty response or bad formatting gracefully
            try:
                structured_data = json.loads(text)
            except json.JSONDecodeError:
                self.log("JSON Decode Error from Gemini. Skipping this batch.")
                return []
            
            final_deals = []
            for item in structured_data:
                idx = item.get('original_id')
                if idx is not None and isinstance(idx, int) and idx < len(scraped_deals):
                    orig = scraped_deals[idx]
                    
                    # Create Deal with the structured data from Gemini
                    # Ensure we preserve the retailer and price from the original scrape
                    final_deals.append(Deal(
                        title=item.get('title', orig.title),
                        price=orig.price,
                        url=orig.url,
                        category=item.get('category', 'Electronics'),
                        features=item.get('features', orig.description),
                        description=item.get('description', orig.description),
                        details=item.get('details', 'N/A'),
                        retailer=orig.retailer # Preserve Retailer info
                    ))
            
            # Limit to 9 items just in case LLM returned more
            return final_deals[:9]

        except Exception as e:
            self.log(f"Gemini Filtering failed: {e}")
            return []

    def scan(self, memory: List[str] = [], query: str = "best tech deals") -> List[Deal]:
        """
        Main entry point: Fetches RSS, Filters duplicates, AI Filters by Query.
        """
        self.log(f"Fetching RSS feeds to find: {query}...")
        raw_scrapes = ScrapedDeal.fetch_deals()
        
        # Deduplicate against memory
        new_scrapes = [d for d in raw_scrapes if d.url not in memory]
        self.log(f"Found {len(raw_scrapes)} items, {len(new_scrapes)} are new.")
        
        if not new_scrapes:
            return []

        # Filter using the specific User Query
        selected_deals = self._filter_deals_with_llm(new_scrapes, query)
        self.log(f"Selected {len(selected_deals)} deals matching '{query}'.")
        
        return selected_deals
