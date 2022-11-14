import numpy as np
import pandas as pd
import janitor  # noqa


def main():
    full = pd.read_csv("data/csv/pst4p.csv", sep=";").clean_names()

    top_parties = get_top_parties(full)
    full["kstrana"] = full["kstrana"].where(lambda x: x.isin(top_parties), -1)
    full["kstrana"] = full["kstrana"].astype(str)
    counts_valid = full.pivot_table(
        "poc_hlasu", "id_okrsky", "kstrana", aggfunc="sum", fill_value=0
    )

    counts_valid.to_parquet("data/counts_valid.pq")

    okrsky = pd.read_csv("data/csv/pst4.csv", sep=";").clean_names()
    okrsky["invalid"] = okrsky["odevz_obal"] - okrsky["pl_hl_celk"]
    okrsky["invalid"] = np.maximum(okrsky["invalid"], 0)
    okrsky[["invalid"]].to_parquet("data/counts_invalid.pq")


def get_top_parties(full: pd.DataFrame) -> list[int]:
    results = (
        full.groupby("kstrana")
        .agg(poc_hlasu=("poc_hlasu", "sum"))
        .sort_values("poc_hlasu", ascending=False)
        .reset_index()
        .assign(share=lambda x: x["poc_hlasu"] / x["poc_hlasu"].sum())
        .loc[lambda x: x["share"] >= 0.01]
    )
    return results["kstrana"].tolist()


if __name__ == '__main__':
    main()
