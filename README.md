# Introduction to Machine Learning using Python Lessons

This is a repository for Sydney Informatics Hub's lesson materials for the "Introduction to Machine Learning using Python" workshop.

## Quickstart

### Installation

Clone the repository:

```bash
$ : git clone https://github.sydney.edu.au/informatics/lessons-mlpy.git
```

Install environment with conda (if you already have a `mlpy` the instructions might not work):

```bash
$ : cd lessons-mlpy
$ : conda env create --file environment.yml
```

### Usage

#### Jupyter Notebooks

Instructions to run the notebooks for the lessons

```bash
$ : cd course/notebooks
$ : conda activate mlpy
(mlpy)$ : jupyter notebook
```

#### Static website

Intructions to run the development website:

```bash
$ : conda activate mlpy
(mlpy)$ : mkdocs serve
```

This will open a website on http://localhost:8001.

The default use the ["Material"](https://squidfunk.github.io/mkdocs-material/) theme but can be switched to normal theme by uncommenting and commenting the following lines in [mkdocs.yml](./mkdocs.yml):

```yaml
theme:
  name: mkdocs
  # name: material
  # features:
  #   - navigation.instant
  #   - navigation.tabs
```

### Structure of Repository

```bash
.
├── course # folder for development with course content and customisations
│   ├── 01-ML1.md
│   ├── fig
│   ├── index.md
│   ├── notebooks # main notebooks for the lessons
│   ├── setup.md
│   └── theme # mkdocs customisations
├── data # data for the notebooks
│   ├── AmesHousingClean.csv
│   ├── AmesHousingDirty.csv
│   ├── breast-cancer-wisconsin.csv
│   └── diabetes.csv
├── docs # mkdocs static website deployment folder
│   ├── 01-ML1
│   ├── 404.html
│   ├── assets
│   ├── css
│   ├── fig
│   ├── index.html
│   ├── notebooks
│   ├── setup
│   ├── sitemap.xml
│   ├── sitemap.xml.gz
│   └── theme
├── environment.yml # conda Python enviroment packages
├── extras # extra content folder
│   ├── 30-RF_knn1.ipynb
│   ├── 50-Classification1.ipynb
│   ├── about.md
│   ├── figs
│   ├── Parameter-Tuning-GBM.md # instructions to tune a GBM model
│   └── Parameter-Tuning-XGBoost.md # instructions to tune a XGBoost model
├── mkdocs.yml # mkdocs setting file
└── README.md
```

The structure of the `notebooks` folder:

```bash
course/notebooks/
├── 03-EDA.ipynb
├── 10-LinReg.ipynb
├── 11-RidgeLassoElasticNet.ipynb
├── 30-RF_knn.ipynb
├── 45-Xgboost.ipynb
├── 50-Classification.ipynb
├── 90-Unsupervised.ipynb
└── data
```
