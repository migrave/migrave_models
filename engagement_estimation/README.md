# Engagement estimation

## Install requirements
```
pip3 install -r requirements.txt
```

## Config
Config file under `config/config.yaml` contains information about what models to train, 
and types of the model (`individualized` or `generalized`), and the dataset.

## Train engagement estimation
The engagement is divided into different levels 
* `low_engagament`: -1 
* `mid_engagement`: 0 
* `high_engagement`: 1

```
python3 train.py --config ./config/config.yaml
```
