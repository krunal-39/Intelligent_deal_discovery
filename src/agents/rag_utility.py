import torch
import faiss
from sentence_transformers import SentenceTransformer
from src.agents.base_agent import Agent
from rag.query_faiss import load_metadata, load_index, TOP_K

class RAGUtility(Agent):
    name = "RAG Utility"
    color = Agent.CYAN

    def __init__(self):
        super().__init__(self.name, self.color)
        try:
            self.metadata = load_metadata("rag/metadata.jsonl")
            self.index = load_index("rag/faiss_index.bin")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = SentenceTransformer("intfloat/e5-large", device=device)
            self.log("✅ RAG System Loaded.")
        except Exception as e:
            self.log(f"❌ RAG Load Failed: {e}")
            self.index = None

    def _get_rag_docs(self, query):
        if not self.index: return []
        q = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, TOP_K)
        docs = []
        for idx in I[0]:
            if idx < 0: continue
            docs.append(self.metadata[idx])
        return docs

    def _build_prompt_string(self, target_data: dict, docs):
        """
        Constructs the prompt EXACTLY matching the training data format.
        target_data: Dict with keys 'title', 'category', 'features', 'description', 'details'
        """
        # 1. System Instruction (Matches training)
        parts = [
            "You are a helpful assistant that estimates the price of a product based on its title, "
            "category, features, description, and details. Return only the numeric price in USD, "
            "formatted with two decimals (e.g., 19.99)."
        ]

        # 2. Few-Shot Examples (Context)
        if docs:
            parts.append("\n\nReference Examples (Context):")
            for i, d in enumerate(docs):
                clean_text = d['prompt'].strip()
                parts.append(f"\n--- Example {i+1} ---\n{clean_text}\nResponse: {d['response']}")

        # 3. Target Product (Matches training format)
        target_prompt = (
            f"\n\nProduct Title: {target_data.get('title', '')}\n"
            f"Product Category: {target_data.get('category', 'General')}\n" 
            f"Product Features: {target_data.get('features', '')}\n"
            f"Product Description: {target_data.get('description', '')}\n"
            f"Additional Details: {target_data.get('details', 'N/A')}\n\n"
            "Predict the price in USD (two decimals)."
        )
        
        parts.append(target_prompt)
        return "\n".join(parts)

    def get_unified_prompt(self, deal_data: dict) -> str:
        """
        Accepts the full deal dictionary now, not just title.
        """
        title = deal_data.get('title', '')
        # Retrieve similar docs using Title (best for semantic search)
        docs = self._get_rag_docs(title)
        return self._build_prompt_string(deal_data, docs)