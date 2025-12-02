import requests
from src.agents.base_agent import Agent

# --- CONFIGURATION ---
DO_CONSOLE = True   
DO_TELEGRAM = True  

class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self):
        super().__init__(self.name, self.color)
        
        # ----- Telegram Setup -----
        self.bot_token = "8112476495:AAFwHs8HnfnRZO5BcwzmPweA8xgMy1uVJrw"
        self.chat_id = "7687044080"
        
        self.telegram_enabled = True
        self.log("✅ Messaging Agent Ready (Batch Mode)")

    def send_batch_summary(self, deal_list: list):
        """
        New method: Takes a LIST of deal dictionaries, 
        combines them into ONE message, and sends it.
        """
        if not deal_list:
            self.log("No deals to send.")
            return

        # 1. Start the message with a Header
        full_message = f"🚀 **GREAT DEALS FOUND ({len(deal_list)})** 🚀\n\n"

        # 2. Loop through every deal and add it to the string
        for deal_data in deal_list:
            deal_text = self._format_rich_message(deal_data)
            full_message += deal_text
            full_message += "\n━━━━━━━━━━━━━━━━━━\n" # Separator line

        # 3. Send the single massive message
        if DO_CONSOLE:
            self.log("\n" + full_message)

        if DO_TELEGRAM:
            self._send_telegram(full_message)

    def _format_rich_message(self, deal_data: dict) -> str:
        """
        Formats a single deal string (used inside the batch loop).
        """
        verdict = deal_data.get('verdict', 'DEAL ALERT')
        icon = "🔥" if deal_data.get('discount_pct', 0) > 20 else "⚠️"

        # Your requested format
        message = (
            f"{icon} --- {verdict} ---\n"
            f"📦 {deal_data.get('title')}\n"
            f"💲 Price: ${deal_data.get('current_price', 0):.2f}\n"
            f"🤑 Discount: ${deal_data.get('discount_amount', 0):.2f} ({deal_data.get('discount_pct', 0):.1f}%)\n"
            f"🔗 {deal_data.get('url')}\n"
        )
        return message

    def _send_telegram(self, message: str):
        if not self.telegram_enabled: return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id, 
            "text": message, 
            "disable_web_page_preview": True # Set True so the chat isn't cluttered with 10 images
        }
        
        try:
            # Telegram has a limit of 4096 chars. If message is too long, we slice it.
            if len(message) > 4000:
                message = message[:4000] + "\n...(truncated due to size limit)"
                payload['text'] = message

            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            self.log(f"Telegram Connection Failed: {e}")