# QRT-Data-Challenge-Football
Repository for the Data Challenge organized by QRT on match results prediction

## Installation
Clone the repo, do a venv, and install the requirements with the following command:
```bash
git clone git@github.com:tboulet/QRT-Data-Challenge-Football.git
cd QRT-Data-Challenge-Football
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then load the data from the ENS Data Challenge website and put it in the `./datas_final/` folder that you will have to create.

## Usage

They is a distinction between two kind of features, from a feature engineering point of view:
- precomputed features: features that are computed once and for all and that are stored in a csv file. They are computed with python scripts in the `./features_engineering/` folder. There loading depends on the config.
- dynamic features: features that are computed on the fly and that are not stored in a csv file. Their computation depends on the config.

There are 3 main components to the project:
- trainer : this is the object that will create and train models using features provided by the features_loader and features_creator.
- loaders : this is the object that will load the features from the csv files.
- creators : this is the object that will create the dynamic features from the original or additional loaded data.

We use cross validation (with K=5 folds) to evaluate the models. The evaluation metric is the accuracy.