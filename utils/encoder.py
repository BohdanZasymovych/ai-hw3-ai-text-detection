"""Fine-tune a transformer classifier and predict with it. Long texts keep only their head and tail tokens.

Not exported from `utils` because it needs torch: `from utils.encoder import ...`.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import LABEL_ORDER
from .evaluation import macro_f1


def gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU: run this step on Kaggle/Colab, or copy its saved artifacts into ARTIFACTS_DIR.")
    return torch.device("cuda")


def _all_gpus(model):
    """Split each batch across all visible GPUs (e.g. 2x T4 on Kaggle); unchanged on a single GPU."""
    return torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model


def head_tail_encode(texts, tokenizer, head: int, tail: int, batch_size: int = 1000) -> dict:
    """Token ids of each text as <s> ... </s>. A text longer than head + tail tokens keeps only its first
    `head` and last `tail` tokens. Also returns the full token count of each text (without special tokens)."""
    ids, n_tokens = [], []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(list(texts[start:start + batch_size]), add_special_tokens=False, verbose=False)["input_ids"]
        for seq in batch:
            n_tokens.append(len(seq))
            if len(seq) > head + tail:
                seq = seq[:head] + seq[len(seq) - tail:]
            ids.append(np.array([tokenizer.cls_token_id, *seq, tokenizer.sep_token_id], dtype=np.int32))
    return {"ids": ids, "n_tokens": np.array(n_tokens)}


def _batch(ids, idx, pad_id, device):
    """Padded input ids and attention mask for the texts at positions `idx`."""
    seqs = [torch.from_numpy(ids[i].astype(np.int64)) for i in idx]
    input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(s) for s in seqs])
    mask = (torch.arange(input_ids.shape[1]) < lengths[:, None]).long()
    return input_ids.to(device), mask.to(device)


def train_encoder(backbone: str, ids, labels, out_dir: Path, *, val_ids, val_labels, lr: float, epochs: int,
                  batch_size: int, warmup: float, weight_decay: float, seed: int, log_every: int = 50) -> dict:
    """Fine-tune `backbone` on token ids and integer labels (positions in LABEL_ORDER): AdamW, linear warmup
    then linear decay, fp16. Saves the model and tokenizer to `out_dir`. Returns the training loss history and
    the validation macro-F1 after each epoch (`val_labels` are label names; used only for monitoring)."""
    device = gpu()
    transformers.logging.set_verbosity_error()  # hides the "newly initialized classifier weights" warning
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=len(LABEL_ORDER), id2label=dict(enumerate(LABEL_ORDER)),
        label2id={label: i for i, label in enumerate(LABEL_ORDER)},
    ).float().to(device)  # fp32 weights; fp16 only inside autocast
    model = _all_gpus(model)

    steps = epochs * math.ceil(len(ids) / batch_size)
    warmup_steps = max(int(warmup * steps), 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / warmup_steps if s < warmup_steps else max(0.0, (steps - s) / (steps - warmup_steps)))
    scaler = torch.amp.GradScaler("cuda")
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    history, losses, step = {"loss": [], "val": []}, [], 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(ids))
        for start in tqdm(range(0, len(order), batch_size), desc=f"epoch {epoch + 1}/{epochs}"):
            idx = order[start:start + batch_size]
            input_ids, mask = _batch(ids, idx, tokenizer.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=mask).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), labels[torch.from_numpy(idx)].to(device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            losses.append(loss.item())
            if len(losses) == log_every:
                history["loss"].append({"step": step, "epoch": epoch + 1, "loss": float(np.mean(losses))})
                losses = []
        model.eval()
        val_pred = np.array(LABEL_ORDER)[_logits(model, val_ids, tokenizer.pad_token_id, device).argmax(axis=1)]
        history["val"].append({"epoch": epoch + 1, "step": step, "macro_f1": macro_f1(val_labels, val_pred)})

    getattr(model, "module", model).save_pretrained(out_dir)  # unwrap DataParallel: a plain HF checkpoint
    tokenizer.save_pretrained(out_dir)
    (Path(out_dir) / "history.json").write_text(json.dumps(history))
    return history


@torch.no_grad()
def _logits(model, ids, pad_id, device, batch_size: int = 64) -> np.ndarray:
    """Logits of a model in eval mode, in input order. Texts are batched by length for speed."""
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = np.zeros((len(ids), len(LABEL_ORDER)), dtype=np.float32)
    for start in tqdm(range(0, len(order), batch_size), desc="predict"):
        idx = order[start:start + batch_size]
        input_ids, mask = _batch(ids, idx, pad_id, device)
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = model(input_ids=input_ids, attention_mask=mask).logits.float().cpu().numpy()
    return out


def predict_logits(model_dir: Path, ids) -> np.ndarray:
    """Logits (columns in LABEL_ORDER) of a saved model for token ids, in input order."""
    if not any(Path(model_dir).iterdir()):
        raise FileNotFoundError(f"No saved model in {model_dir}: train it first or copy it from the Kaggle run.")
    device = gpu()
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().to(device).eval()
    assert [model.config.id2label[i] for i in range(len(LABEL_ORDER))] == LABEL_ORDER, "label order differs"
    return _logits(_all_gpus(model), ids, model.config.pad_token_id, device)
