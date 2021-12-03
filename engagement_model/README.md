# Engagement estimation

## Install requirements
```
pip3 install -r requirements.txt
```

## Config
Config file under `config/config.yaml` contains information about what models to train, and types of the model (`individualized` or `generalized`), and dataset file.

## Dataset
The dataset is based on [this paper](https://www.science.org/doi/10.1126/scirobotics.aaz3791) and `migrave_all.csv` contains visual features of four MigrAVE staffs.

## Train engagement estimation
```
python3 train.py --config ./config/config.yaml
```
