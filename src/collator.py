# src/collator.py
import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorForPricePrediction:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Detect whether batch has pre-tokenized data
        pretokenized = "input_ids" in batch[0]

        if pretokenized:
            # Already tokenized dataset (input_ids, labels, etc.)
            input_ids_list = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
            labels_list = []
            for ids in input_ids_list:
                # create dummy labels same as input (for LM)
                lbl = ids.clone()
                lbl[: -1] = -100  # mask all but last token 
                labels_list.append(lbl)

            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
            attention_mask = (input_ids != pad_id).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # Otherwise, fall back to prompt/response processing
        prompts = [ex["prompt"] for ex in batch]
        responses = [str(ex["response"]) for ex in batch]

        enc_prompts = [
            self.tokenizer(p, truncation=True, max_length=self.max_length, add_special_tokens=False)
            for p in prompts
        ]
        enc_responses = [
            self.tokenizer(r, truncation=True, max_length=128, add_special_tokens=False)
            for r in responses
        ]

        input_ids_list, labels_list = [], []
        for enc_p, enc_r in zip(enc_prompts, enc_responses):
            p_ids, r_ids = enc_p["input_ids"], enc_r["input_ids"]
            full_ids = p_ids + r_ids
            lbl = torch.full((len(full_ids),), fill_value=-100, dtype=torch.long)
            if len(r_ids) > 0:
                lbl[-len(r_ids):] = torch.tensor(r_ids, dtype=torch.long)
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(lbl)

        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
