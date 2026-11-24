"""Token surprise of texts under an open causal LM (used only to score texts, never trained), and per-text features.

Surprise of token i: s_i = -log p(t_i | t_1..t_{i-1}). Perplexity is exp(mean s_i).
Not exported from `utils` because it needs torch: `from utils.surprise import ...`.
"""
import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .encoder import _batch, all_gpus, gpu

FEATURES = ["surprise mean", "surprise std", "chunk spread", "chunk min", "chunk max"]


class _Surprise(torch.nn.Module):
    """Per-token surprise for a batch. The vocabulary logits (~150k per token) are computed in slices of positions,
    so they never all sit in GPU memory at once; only the small surprise matrix is returned (and gathered)."""

    def __init__(self, lm, slice_len: int = 256):
        super().__init__()
        self.body, self.head, self.slice_len = lm.base_model, lm.get_output_embeddings(), slice_len

    def forward(self, input_ids, mask):
        hidden = self.body(input_ids=input_ids, attention_mask=mask).last_hidden_state[:, :-1]
        target = input_ids[:, 1:]
        out = []
        for s in range(0, hidden.shape[1], self.slice_len):
            logits = self.head(hidden[:, s:s + self.slice_len]).float()
            out.append(torch.nn.functional.cross_entropy(logits.transpose(1, 2), target[:, s:s + self.slice_len],
                                                         reduction="none"))
        return torch.cat(out, dim=1)


def token_surprise(texts, model_name: str, max_tokens: int = 2048, batch_size: int = 16) -> list:
    """Surprise of every token after the first (float32 array per text) for the first `max_tokens` tokens."""
    device = gpu()
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ids = []
    for start in range(0, len(texts), 1000):
        batch = tokenizer(list(texts[start:start + 1000]), add_special_tokens=False, truncation=True,
                          max_length=max_tokens, verbose=False)["input_ids"]
        ids += [np.array(seq, dtype=np.int32) for seq in batch]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(model_name).float().to(device).eval()  # fp16 only inside autocast
    scorer = all_gpus(_Surprise(lm))

    order = np.argsort([len(x) for x in ids], kind="stable")
    out = [None] * len(ids)
    with torch.no_grad():
        for start in tqdm(range(0, len(order), batch_size), desc="surprise"):
            idx = order[start:start + batch_size]
            input_ids, mask = _batch(ids, idx, pad_id, device)
            with torch.autocast("cuda", dtype=torch.float16):
                surprise = scorer(input_ids, mask).float().cpu().numpy()
            for row, i in enumerate(idx):
                out[i] = surprise[row, :max(len(ids[i]) - 1, 0)].astype(np.float32)
    return out


def surprise_features(surprise: list, chunk: int = 128) -> pd.DataFrame:
    """Per text: mean and std of the token surprise, and the spread (std), minimum and maximum of the mean
    surprise of consecutive chunks of about `chunk` tokens (a text shorter than `chunk` is one chunk)."""
    rows = []
    for s in surprise:
        if len(s) == 0:
            rows.append([np.nan] * len(FEATURES))
            continue
        means = [c.mean() for c in np.array_split(s, max(len(s) // chunk, 1))]
        rows.append([s.mean(), s.std(), np.std(means), min(means), max(means)])
    return pd.DataFrame(rows, columns=FEATURES)
