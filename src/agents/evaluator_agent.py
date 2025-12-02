from src.agents.base_agent import Agent

class EvaluatorAgent(Agent):
    """
    Analyzes the gap between Real Price and AI Estimated Price.
    Decides if a deal is worth alerting based ONLY on percentage discount.
    """
    name = "Evaluator Agent"
    color = Agent.MAGENTA

    # Threshold for triggering an alert 
    DISCOUNT_THRESHOLD_PERCENT = 10.0

    def __init__(self):
        super().__init__(self.name, self.color)
        self.log("Ready to evaluate deals.")

    def evaluate(self, current_price: float, ai_estimated_price: float) -> dict:
        """
        Compares actual price vs AI price.
        Returns a dictionary containing the decision signal ('is_deal').
        """
        # 1. Calculate Metrics
        discount_amount = ai_estimated_price - current_price
        
        if ai_estimated_price > 0:
            discount_pct = (discount_amount / ai_estimated_price) * 100
        else:
            discount_pct = 0.0

        # 2. Determine Verdict 
        verdict = "PASS"
        is_alert_worthy = False

        if discount_pct >= self.DISCOUNT_THRESHOLD_PERCENT:
            verdict = "GREAT DEAL"
            is_alert_worthy = True
        elif discount_pct > 0:
            verdict = "PASS (Small Discount)"
            is_alert_worthy = False
        else:
            verdict = "OVERPRICED"

        # 3. Log the decision
        self.log(f"Analysis: Real ${current_price} vs AI ${ai_estimated_price} -> {verdict} ({discount_pct:.1f}%)")

        # 4. Return the Signal package
        return {
            "is_deal": is_alert_worthy,
            "verdict": verdict,
            "discount_amount": round(discount_amount, 2),
            "discount_pct": round(discount_pct, 2),
            "ai_price": ai_estimated_price,
            "real_price": current_price
        }
