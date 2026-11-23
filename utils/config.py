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
PPL_MODEL = "Qwen/Qwen2.5-0.5B"  # open causal LM used only to score texts (perplexity features)

# Fixed colours of the models in comparison plots.
MODEL_COLORS = {
    "baseline": "#9D9D9D", "baseline 100k": "#5E5E5E",
    "encoder": "#4C78A8", "encoder chunked": "#72B7B2", "encoder large": "#1F3B73",
    "stacker": "#B279A2", "final stacker": "#9D755D", "stacker windows": "#54A24B", "stacker + perplexity": "#E45756", "perplexity only": "#F2A93B",
}
