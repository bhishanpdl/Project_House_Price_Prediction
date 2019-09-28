# Cookiecutter Data Science
Template taken from:  https://github.com/drivendata/cookiecutter-data-science
Medium post about cookiecutter: https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e


# Install your custom module src
```bash
ls # You must be in project root dierectory where there is src folder.
which pip # /Users/poudel/miniconda3/envs/dataSc/bin/pip
/Users/poudel/miniconda3/envs/dataSc/bin/pip install --editable .
```

# Use your custom module src in notebooks
```bash
# in the folder src create folder funct and a file Foo.py
cd src
mkdir funct
cp __init__.py funct/  # we need to create init file

# create new functions
vi Foo.py # we can also use vscode editor

def foo(x):
    if x is 'foo':
        return 'foo'
    else:
        return 'bar'

pwd # something/src/funct
ls # __init__.py, Foo.py
```
```python
# Now to go to notebooks directory and create or open a notebook
# keep this so that when we modify functions it will reflect here.
%load_ext autoreload
%autoreload 2

# my custom functions
from src.funct import Foo

print(Foo.foo('hi')) # prints bar
```

# Project Diectory Structure
------------

The directory structure of your new project looks like this:

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```

# Data is immutable
Don't ever edit your raw data, especially not manually, and especially not in Excel. Don't overwrite your raw data. Don't save multiple versions of the raw data. Treat the data (and its format) as immutable. The code you write should move the raw data through a pipeline to your final analysis. You shouldn't have to run all of the steps every time you want to make a new figure (see Analysis is a DAG), but anyone should be able to reproduce the final products with only the code in src and the data in data/raw.

Also, if data is immutable, it doesn't need source control in the same way that code does. Therefore, by default, the data folder is included in the .gitignore file. If you have a small amount of data that rarely changes, you may want to include the data in the repository. Github currently warns if files are over 50MB and rejects files over 100MB. Some other options for storing/syncing large data include AWS S3 with a syncing tool (e.g., s3cmd), Git Large File Storage, Git Annex, and dat. Currently by default, we ask for an S3 bucket and use AWS CLI to sync data in the data folder with the server.

# Notebooks are for exploration and communication
Notebook packages like the Jupyter notebook, Beaker notebook, Zeppelin, and other literate programming tools are very effective for exploratory data analysis. However, these tools can be less effective for reproducing an analysis. When we use notebooks in our work, we often subdivide the notebooks folder. For example, notebooks/exploratory contains initial explorations, whereas notebooks/reports is more polished work that can be exported as html to the reports directory.

Since notebooks are challenging objects for source control (e.g., diffs of the json are often not human-readable and merging is near impossible), we recommended not collaborating directly with others on Jupyter notebooks. There are two steps we recommend for using notebooks effectively:

Follow a naming convention that shows the owner and the order the analysis was done in. We use the format <step>-<ghuser>-<description>.ipynb (e.g., 0.3-bull-visualize-distributions.ipynb).

Refactor the good parts. Don't write code to do the same task in multiple notebooks. If it's a data preprocessing task, put it in the pipeline at src/data/make_dataset.py and load data from data/interim. If it's useful utility code, refactor it to src.

Now by default we turn the project into a Python package (see the setup.py file). You can import your code and use it in notebooks with a cell like the following:
```python
# OPTIONAL: Load the "autoreload" extension so that code can change
%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
%autoreload 2

from src.data import make_dataset
```

# Analysis is a DAG 
- [Directed acyclic graph](https://www.wikiwand.com/en/Directed_acyclic_graph)
Often in an analysis you have long-running steps that preprocess data or train models. If these steps have been run already (and you have stored the output somewhere like the data/interim directory), you don't want to wait to rerun them every time. We prefer make for managing steps that depend on each other, especially the long-running ones. Make is a common tool on Unix-based platforms (and is available for Windows). Following the make documentation, Makefile conventions, and portability guide will help ensure your Makefiles work effectively across systems. Here are some examples to get started. A number of data folks use make as their tool of choice, including Mike Bostock.

There are other tools for managing DAGs that are written in Python instead of a DSL (e.g., Paver, Luigi, Airflow, Snakemake, Ruffus, or Joblib). Feel free to use these if they are more appropriate for your analysis.

# Build from the environment up
The first step in reproducing an analysis is always reproducing the computational environment it was run in. You need the same tools, the same libraries, and the same versions to make everything play nicely together.

One effective approach to this is use virtualenv (we recommend virtualenvwrapper for managing virtualenvs). By listing all of your requirements in the repository (we include a requirements.txt file) you can easily track the packages needed to recreate the analysis. Here is a good workflow:

- Run mkvirtualenv when creating a new project
- pip install the packages that your analysis needs
- Run pip freeze > requirements.txt to pin the exact package versions used to recreate the analysis
- If you find you need to install another package, run pip freeze > requirements.txt again and commit the changes to version control.


If you have more complex requirements for recreating your environment, consider a virtual machine based approach such as Docker or Vagrant. Both of these tools use text-based formats (Dockerfile and Vagrantfile respectively) you can easily add to source control to describe how to create a virtual machine with the requirements you need.


