"""Settings shared by all notebooks. Changing one recomputes every cached step whose key uses it."""
SEED = 42

# Fixed data used by every model comparison.
VAL_FRAC = 0.1  # share of each (language, domain, label) cell held out for validation
SUBSAMPLE_SIZE = 50_000  # training texts per model
SUBSAMPLE_FLOOR = 2_000  # cells smaller than this are kept in full

LABEL_ORDER = ["human", "ai", "fusion"]
LABEL_COLORS = {"human": "#4C78A8", "ai": "#E45756", "fusion": "#F2A93B"}

ENCODER = "xlm-roberta-base"
