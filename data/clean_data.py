import pandas as pd

# --- 1. load --------------------------------------------------------------
SRC = "data/detective-puzzles.csv"
df = pd.read_csv(SRC)

# --- 2. detect murder mysteries ------------------------------------------
# Feel free to adjust / extend these keywords to tighten or loosen the filter.
KEYWORDS = [
    "murder", "murdered", "killer", "kill", "killed",
    "homicide", "slain", "corpse"
]

def is_murder_mystery(row: pd.Series) -> bool:
    """Return True if any keyword appears in the case title, text or outcome."""
    text = " ".join(
        str(row[c]).lower()
        for c in ("case_name", "mystery_text", "outcome")
        if c in row
    )
    return any(kw in text for kw in KEYWORDS)

df["is_murder_mystery"] = df.apply(is_murder_mystery, axis=1)

# --- 3. keep only the murder‑mystery rows ---------------------------------
murder_df = df[df["is_murder_mystery"]].drop(columns="is_murder_mystery")

# --- 4. save --------------------------------------------------------------
DST = "data/murder-mysteries.csv"
murder_df.to_csv(DST, index=False)

print(
    f"Cleaned CSV written to {DST}\n"
    f"Original rows : {len(df)}\n"
    f"Murder rows   : {len(murder_df)}"
)
