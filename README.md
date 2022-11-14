#

### Get data

```bash
wget https://volby.cz/opendata/ps2021/PS2021_data_20211010_csv.zip -P data

```

### Setup environment

```bash
poetry shell
poetry install
```

### Run

The scripts do not have arguments exposed. You need to change those in code.

```bash
python prepare_data.py
python mcmc.py
python collect_results.py
```

### Podminene rozdeleni pravdepodobnosti:

- platne hlasy per strana + neplatne hlasy per strana => posterior pravdepodobnosti pro neplatny hlas pro stranu
- platne hlasy pro stranu v okrsku + neplatne hlasy pro stranu v okrsku => posterior pravdepodobnosti pro stranu v okrsku
- sampling stran pro neplatne
  - posterior pro neplatny => sample pravdepodobnost pro neplatny
  - posterior pro stranu v okrsku => sample pravdepodobnosti pro stranu v okrsku
  - dopocitat pravdepodobnost pro stranu pro neplatny hlas
  - sample z pravdepodobnosti