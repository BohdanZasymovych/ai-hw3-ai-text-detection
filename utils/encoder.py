"""Fine-tune a transformer classifier and predict with it. Training uses head + tail truncation of long texts;
prediction uses either the same head + tail or several windows (chunks) per text.

Not exported from `utils` because it needs torch: `from utils.encoder import ...`.
"""
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .cache import _norm_key
from .config import LABEL_ORDER
from .evaluation import macro_f1
from .paths import ARTIFACTS_DIR, EXTRA_ARTIFACT_DIRS


def gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU: run this step on Kaggle/Colab, or copy its saved artifacts into ARTIFACTS_DIR.")
    return torch.device("cuda")


def all_gpus(model):
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


def chunk_encode(texts, tokenizer, window: int = 510, max_chunks: int = 8, batch_size: int = 1000) -> dict:
    """Windows of `window` tokens, each as <s> ... </s>. A text that fits is one window; a longer one gets
    ceil(n / window) windows (at most `max_chunks`), spread evenly from its first to its last token.
    Returns the windows, the text (`doc`) and position (`pos`) of each window, and the windows per text."""
    ids, doc, pos, n_windows = [], [], [], []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(list(texts[start:start + batch_size]), add_special_tokens=False, verbose=False)["input_ids"]
        for j, seq in enumerate(batch):
            k = max(1, min(math.ceil(len(seq) / window), max_chunks))
            offsets = np.linspace(0, max(len(seq) - window, 0), k).round().astype(int)
            for p, offset in enumerate(offsets):
                ids.append(np.array([tokenizer.cls_token_id, *seq[offset:offset + window], tokenizer.sep_token_id],
                                    dtype=np.int32))
                doc.append(start + j)
                pos.append(p)
            n_windows.append(k)
    return {"ids": ids, "doc": np.array(doc), "pos": np.array(pos), "n_windows": np.array(n_windows)}


def combine_chunks(logits: np.ndarray, doc: np.ndarray, n_docs: int, method: str) -> np.ndarray:
    """One row of class scores per text from the logits of its windows:
    "mean_logits" (average), "mean_probs" (log of the average probability) or
    "max_fusion" (average, but the fusion score is the highest over the windows)."""
    count = np.bincount(doc, minlength=n_docs)[:, None]
    out = np.zeros((n_docs, logits.shape[1]))
    if method == "mean_probs":
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        np.add.at(out, doc, probs / probs.sum(axis=1, keepdims=True))
        return np.log(out / count)
    np.add.at(out, doc, logits)
    out /= count
    if method == "max_fusion":
        fusion = LABEL_ORDER.index("fusion")
        top = np.full(n_docs, -np.inf)
        np.maximum.at(top, doc, logits[:, fusion])
        out[:, fusion] = top
    elif method != "mean_logits":
        raise ValueError(f"unknown method {method}")
    return out


def _batch(ids, idx, pad_id, device):
    """Padded input ids and attention mask for the texts at positions `idx`."""
    seqs = [torch.from_numpy(ids[i].astype(np.int64)) for i in idx]
    input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(s) for s in seqs])
    mask = (torch.arange(input_ids.shape[1]) < lengths[:, None]).long()
    return input_ids.to(device), mask.to(device)


def _load_checkpoint(name: str, key: dict):
    """The latest end-of-epoch training state saved under `name` for this `key` (here or in a Kaggle input)."""
    for base in (ARTIFACTS_DIR, *EXTRA_ARTIFACT_DIRS):
        path = base / name / "state.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["key"] == _norm_key(key):
                return state
    return None


def _save_checkpoint(name: str, state: dict) -> None:
    path = ARTIFACTS_DIR / name / "state.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)  # write then rename: no half-written checkpoint


def train_encoder(backbone: str, ids, labels, out_dir: Path, *, val_ids, val_labels, lr: float, epochs: int,
                  batch_size: int, warmup: float, weight_decay: float, seed: int, grad_accum: int = 1,
                  checkpoint: str | None = None, key: dict | None = None, log_every: int = 50) -> dict:
    """Fine-tune `backbone` on token ids and integer labels (positions in LABEL_ORDER): AdamW, linear warmup
    then linear decay, fp16. Each batch of `batch_size` texts is processed in `grad_accum` smaller parts
    (same result, less GPU memory). Saves the model and tokenizer to `out_dir`. Returns the training loss history
    and the validation macro-F1 after each epoch (`val_labels` are label names; used only for monitoring).

    With `checkpoint` (a name) and `key`, the full training state is saved after every epoch and a run with the
    same key resumes from it, e.g. after a Kaggle session ended; the checkpoint is deleted when training ends."""
    device = gpu()
    transformers.logging.set_verbosity_error()  # hides the "newly initialized classifier weights" warning
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    core = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=len(LABEL_ORDER), id2label=dict(enumerate(LABEL_ORDER)),
        label2id={label: i for i, label in enumerate(LABEL_ORDER)},
    ).float().to(device)  # fp32 weights; fp16 only inside autocast

    steps = epochs * math.ceil(len(ids) / batch_size)
    warmup_steps = max(int(warmup * steps), 1)
    optimizer = torch.optim.AdamW(core.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / warmup_steps if s < warmup_steps else max(0.0, (steps - s) / (steps - warmup_steps)))
    scaler = torch.amp.GradScaler("cuda")
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    history, losses, step, first_epoch = {"loss": [], "val": []}, [], 0, 0
    state = _load_checkpoint(checkpoint, key) if checkpoint else None
    if state is not None:
        core.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["rng"]
        history, step, first_epoch = state["history"], state["step"], state["epoch"]
        print(f"[checkpoint] resumed after epoch {first_epoch}")
    model = all_gpus(core)

    for epoch in range(first_epoch, epochs):
        model.train()
        order = rng.permutation(len(ids))
        for start in tqdm(range(0, len(order), batch_size), desc=f"epoch {epoch + 1}/{epochs}"):
            idx = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for part in np.array_split(idx, grad_accum):
                if len(part) == 0:
                    continue
                input_ids, mask = _batch(ids, part, tokenizer.pad_token_id, device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=input_ids, attention_mask=mask).logits
                target = labels[torch.from_numpy(part)].to(device)
                loss = torch.nn.functional.cross_entropy(logits.float(), target) * (len(part) / len(idx))
                scaler.scale(loss).backward()
                batch_loss += loss.item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            losses.append(batch_loss)
            if len(losses) == log_every:
                history["loss"].append({"step": step, "epoch": epoch + 1, "loss": float(np.mean(losses))})
                losses = []
        model.eval()
        val_pred = np.array(LABEL_ORDER)[_logits(model, val_ids, tokenizer.pad_token_id, device).argmax(axis=1)]
        history["val"].append({"epoch": epoch + 1, "step": step, "macro_f1": macro_f1(val_labels, val_pred)})
        if checkpoint and epoch + 1 < epochs:
            _save_checkpoint(checkpoint, {
                "key": _norm_key(key), "epoch": epoch + 1, "step": step, "history": history,
                "rng": rng.bit_generator.state, "model": core.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            })

    core.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (Path(out_dir) / "history.json").write_text(json.dumps(history))
    if checkpoint:
        shutil.rmtree(ARTIFACTS_DIR / checkpoint, ignore_errors=True)
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


def predict_logits(model_dir: Path | None, ids) -> np.ndarray:
    """Logits (columns in LABEL_ORDER) of a saved model for token ids, in input order."""
    if model_dir is None or not any(Path(model_dir).iterdir()):
        raise FileNotFoundError(f"No saved model in {model_dir}: train it first or attach the run that saved it.")
    device = gpu()
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().to(device).eval()
    assert [model.config.id2label[i] for i in range(len(LABEL_ORDER))] == LABEL_ORDER, "label order differs"
    return _logits(all_gpus(model), ids, model.config.pad_token_id, device)
