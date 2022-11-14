import json
import os

import numpy as np


party_names = [
    "Ostatni",
    "VOLNÝ blok",
    "Svoboda a př. demokracie (SPD)",
    "Česká str.sociálně demokrat.",
    "Trikolora Svobodní Soukromníci",
    "PŘÍSAHA Roberta Šlachty",
    "SPOLU – ODS, KDU-ČSL, TOP 09",
    "PIRÁTI a STAROSTOVÉ",
    "Komunistická str.Čech a Moravy",
    "ANO 2011",
]


def main():
    all_results = []
    for file_name in os.listdir("results"):
        if not file_name.endswith(".json"):
            continue
        with open(f"results/{file_name}") as f:
            res = json.load(f)
            all_results.append(res)

    print(f"{len(all_results)} results")
    if not all_results:
        return
    x = np.array(all_results).mean(0)

    for party, prob in zip(party_names, x):
        print(f"{party:>30}: {prob:.2%}")


if __name__ == '__main__':
    main()
