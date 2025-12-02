
from pydantic import BaseModel
from typing import List, Optional
from bs4 import BeautifulSoup
import re
import feedparser
import time

# Combined Feed List: Slickdeals + DealNews
RSS_FEEDS = [
    
     "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
     "https://www.dealnews.com/c142/Electronics/?rss=1",
    
    
    # --- DealNews (Computers & Laptops) ---
     "https://www.dealnews.com/c49/Computers/Laptops/?rss=1",
     "https://www.dealnews.com/c49/Computers/Laptops/f31/Gaming/?rss=1",
     "https://www.dealnews.com/c41/Computers/Apple-Computers/?rss=1",
    
    # --- Slickdeals (Custom Searches) ---
    "https://slickdeals.net/newsearch.php?mode=popdeals&searcharea=deals&searchin=first&rss=1&q=computers",
    "https://slickdeals.net/newsearch.php?mode=popdeals&searcharea=deals&searchin=first&rss=1&q=laptop",
    "https://slickdeals.net/newsearch.php?mode=popdeals&searcharea=deals&searchin=first&rss=1&q=monitor+mouse+keyboard"
]

def clean_html(html_snippet: str) -> str:
    """
    Cleans HTML tags. 
    Uses ' | ' separator to keep features distinct.
    """
    if not html_snippet: return ""
    soup = BeautifulSoup(html_snippet, 'html.parser')
    
    # Slickdeals specific cleanup: Remove the "Thumb Score" div
    for div in soup.find_all("div"):
        if "Thumb Score" in div.get_text():
            div.decompose()

    return soup.get_text(separator=' | ', strip=True)

class Deal(BaseModel):
    """
    Represents a processed deal ready for pricing.
    """
    title: str
    price: float
    url: str
    category: str = "General"
    features: str = ""
    description: str
    details: str = "N/A"
    retailer: str = "Unknown"

class ScrapedDeal:
    """Represents a raw item from the RSS feed."""
    def __init__(self, entry):
        self.title = entry.get('title', 'Unknown Deal')
        self.url = entry.get('link', '')
        
        # 1. Handle different RSS content structures
        # Slickdeals uses 'content' -> 'encoded', DealNews uses 'summary' -> 'description'
        if hasattr(entry, 'content') and len(entry.content) > 0:
            raw_html = entry.content[0].value
        else:
            raw_html = entry.get('summary', '') or entry.get('description', '')

        # 2. Extract Details
        self.retailer = self._extract_retailer(raw_html)
        self.description = clean_html(raw_html)
        
        # 3. Extract Price (STRICT: Title Only)
        # We only pass the TITLE to the price extractor.
        self.price = self._extract_price(self.title)

    def _extract_price(self, text):
        """
        Extracts price strictly from the provided text (Title).
        Ignores amounts related to savings (e.g. '$600 off').
        """
        # Regex Breakdown:
        # \$             -> Look for dollar sign
        # (\d...)        -> Capture the number (e.g. 1,234.56)
        # (?!...)        -> Negative Lookahead: Fail if followed by "off", "discount", etc.
        #                   This prevents capturing "Up to $600 off" as the price.
        
        matches = re.findall(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(?!\s*(?:off|less|discount|savings|rebate))', text, re.IGNORECASE)
        
        if matches:
            # Return the last match found (usually the final sale price in "Was $100, Now $50")
            val = matches[-1]
            return float(val.replace(',', ''))
            
        return 0.0

    def _extract_retailer(self, html_text):
        """Attempts to identify the store."""
        # 1. Try Regex: "Shop/Buy Now at [Store]"
        text_match = re.search(r'(?:Shop|Buy) Now at (.+?)(?:$|\.|\s{2,}|<)', html_text, re.IGNORECASE)
        if text_match:
            return text_match.group(1).strip()
            
        # 2. Try Slickdeals Metadata: "Sold and shipped by [Store]"
        sold_match = re.search(r'Sold and shipped by\s+([\w\s]+)', html_text, re.IGNORECASE)
        if sold_match:
            return sold_match.group(1).strip()

        return "Unknown"

    @classmethod
    def fetch_deals(cls) -> List['ScrapedDeal']:
        """Fetches and parses deals from all configured RSS feeds."""
        found_deals = []
        for url in RSS_FEEDS:
            try:
                # Use a browser User-Agent to avoid being blocked
                feed = feedparser.parse(url, agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
                
                for entry in feed.entries[:6]: 
                    deal = cls(entry)
                    # REJECTION LOGIC: Only keep deal if we found a valid price in the title
                    if 0 < deal.price < 999.00:
                        found_deals.append(deal)
                        
                time.sleep(0.5) 
            except Exception as e:
                print(f" RSS Error ({url}): {e}")
        return found_deals
    
