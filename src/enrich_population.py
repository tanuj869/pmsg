"""Join official state population estimates and add per-capita adoption rates."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def build_population_panel() -> pd.DataFrame:
    model = pd.read_csv(PROCESSED / "modeling_table.csv")
    population = pd.read_csv(RAW / "state_population_mohfw_2021.csv")
    population["state"] = population["state"].replace({"Delhi": "NCT of Delhi"})
    # Keep every population-source State/UT, even where the scheme snapshot is
    # absent. A blank adoption rate makes source coverage visible rather than
    # silently shrinking the canonical jurisdiction list.
    panel = model.merge(population, on="state", how="right", validate="one_to_one")
    if panel["population_2021"].isna().any():
        missing = panel.loc[panel["population_2021"].isna(), "state"].tolist()
        raise ValueError(f"Population estimates missing for states: {missing}")
    panel["adoption_data_available"] = panel["installations_aug2026"].notna()
    panel["installations_per_lakh_population"] = (
        panel["installations_aug2026"] / panel["population_2021"] * 100_000
    ).round(3)
    panel["applications_per_lakh_population"] = (
        panel["applications_aug2026"] / panel["population_2021"] * 100_000
    ).round(3)
    panel["population_denominator_year"] = 2021
    panel["adoption_numerator_date"] = "2026-08-23"
    out = PROCESSED / "modeling_table_population_normalized.csv"
    panel.to_csv(out, index=False)
    return panel


if __name__ == "__main__":
    result = build_population_panel()
    print(f"Saved {len(result)} state/UT rows with population-normalized rates.")
