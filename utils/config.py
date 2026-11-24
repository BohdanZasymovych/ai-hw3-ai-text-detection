"""Settings shared by all notebooks. Changing one recomputes every cached step whose key uses it."""
SEED = 42

# Fixed data used by every model comparison.
VAL_FRAC = 0.1  # share of each (language, domain, label) cell held out for validation
SUBSAMPLE_SIZE = 50_000  # training texts per model
SUBSAMPLE_FLOOR = 2_000  # cells smaller than this are kept in full

LABEL_ORDER = ["human", "ai", "fusion"]
LABEL_COLORS = {"human": "#4C78A8", "ai": "#E45756", "fusion": "#F2A93B"}

ENCODER = "xlm-roberta-base"
LARGE_ENCODER = "xlm-roberta-large"
SURPRISE_MODEL = "Qwen/Qwen2.5-0.5B"  # open causal LM used only to score texts (surprise features)

# Fixed colours of the models in comparison plots.
MODEL_COLORS = {
    "TF-IDF 50k": "#9D9D9D", "TF-IDF 100k": "#5E5E5E",
    "XLM-R base 50k": "#4C78A8", "XLM-R base 50k, window mean": "#72B7B2", "XLM-R large 100k": "#1F3B73",
    "Stack: XLM-R base": "#B279A2", "Stack: XLM-R base + windows": "#54A24B",
    "Stack: XLM-R base + surprise": "#E45756", "Stack: surprise": "#F2A93B",
    "Stack: XLM-R large": "#C8A2C8", "Stack: XLM-R large + base windows": "#8CD17D",
    "Stack: XLM-R large + large windows": "#59A14F", "Stack: XLM-R large + both windows": "#2E7D32",
    "Stack: final": "#9D755D",
}
