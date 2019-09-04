# Great Expectations Pipeline Tests for scikit-learn

This repository is an example for how the [Great Expectations](https://github.com/great-expectations/great_expectations) 
library could be integrated within a scikit-learn machine learning pipeline to ensure that data 
inputs, transformed data, and even the model predictions conform to an expected standard. 

## Table of Contents

 - [Background](#background)
 - [Project Folder Structure](#project-folder-structure)
 - [scikit-learn Pipeline](#scikit-learn-pipeline)
 - [Creating Expectations](#creating-expectations)
 - [Running Your Analysis](#running-your-analysis) 
 - [Demo Scenarios](#demo-scenarios)  


---

### Background

In this example assume that we want to build a machine learning pipeline to 
estimate the weight of two species of birds. Along with the species we have other 
attributes such as, color, beak_ratio, claw_length, and wing_density. The raw data 
is available at [data/raw-data.csv](./data/raw-data.csv).Please note that this is fake 
data that was generated using the script [lib/datagenerator.py](./lib/datagenerator.py). 
The purpose of using a fake dataset was to have a small reproducible example.

The examples below assume you have installed Python along with the libraries **numpy**, 
**pandas**, **scikit-learn**, and **great_expectations**.


---

### Project Folder Structure

This project contains five folders, one README, and one file entitled `main.py`. 
In this structure we have a `data` folder and an `output` folder to physically separate 
the inputs and outputs:  

1.  [`data`](./data): Folder for raw, unprocessed data
2.  [`output`](./output): Folder for cleaned data, plots, etc.

There are three other folders in the structure:

3. [`lib`](./lib): Folder for Python scripts that hold global settings and functions
4. [`great_expectations`](./great_expectations): Folder for configurations and artifacts of pipeline tests
5. [`scenarios`](./scenarios): Folder for holding artifacts that create demo scenarios

You should also notice at the top-level of the project folder there are the two files: 

6. [`main.py`](main.py): A single Python file that runs the main analysis for the project 
7. [`README.md`](README.md): A file that explains what this project is about

Using this folder configuration we have a single Python file ([`main.py`](main.py)) that runs 
our entire analysis but interacts with the other folders to read and write artifacts 
of the analysis. The script contains code that validates data during the run against 
expectations to ensure the integrity of the analysis.


---

### scikit-learn Pipeline

In **scikit-learn** there is functionality where you can take multiple "transformers" 
and chain them together to preprocess data and model it. The [`main.py`](main.py) 
file is where these transformers preprocess the raw data located in [`data/raw-data.csv`](./data/raw-data.csv). 
Here we will introduce the Great Expectations (`ge`) library in three key areas to 
validate assumptions about data in our pipeline: 

1. Use `ge` to check the raw data
2. Use `ge` to check the data after preprocessing, which is right before modeling
3. Use `ge` to check the difference between predicted and actuals in a holdout dataset

First, it is pretty obvious that Great Expectations could be used to validate the input, raw data. 
The phrase "garbage in, garbage out" applies here. It does not matter how good our 
pipeline is if we put bad data into it, so we prevent that from happening.

Second, checking preprocessed data is also important. This ensures that all of your transformers 
behaved as you would expect in case they encoded data incorrectly or changed in another 
environment if you pickle them after fitting them to data, etc. Even another analyst 
could tweak a transformer parameter in `main.py` which is running the analysis and 
the pipeline tests, if written correctly, should catch the change in parameters of 
the transformers. This is extremely helpful in ensuring the data you are modeling 
is what you expect.

Third, we check the model errors on a holdout set. This is also important in ensuring 
that there are no extremely large errors caused by drift in the inputs, outliers, or a 
decline in model performance.


---

### Creating Expectations

To implement the three types of validation described above we started by initializing a 
Great Expectations project by running the command below in the terminal. 

**NOTE**: If you cloned this repo and are following along you can delete the 
`great_expectations` folder and follow the instructions below. We assume you are 
setting this up in a project where Great Expectations has not already been initialized.

```
great_expectations init
```

In the initialization prompts we declined to add a DataSource. By declining the DataSource 
configuration we were able to quickly and consistently setup multiple DataSources by 
running the script [`set-datasources`](./great_expectations/notebooks/set-datasources.py). 
You can always check which DataAssets are available from DataSources by running the commands below 
in Python. This should show the three DataAssets from the `.csv` files: 1) `raw-data`, 
2) `modeling-data`, and 3) `holdout-error-data`.  

```
import great_expectations as ge
import great_expectations.jupyter_ux
context = ge.data_context.DataContext()
great_expectations.jupyter_ux.list_available_data_asset_names(context)

data_source: data__dir (pandas)
  generator_name: default (subdir_reader)
    generator_asset: raw-data
data_source: output__dir (pandas)
  generator_name: default (subdir_reader)
    generator_asset: modeling-data
    generator_asset: holdout-error-data
```

The next step is creating expectations for these three DataAssets. The expectations 
we used were created by the following scripts: 

1. Raw data: [./great_expectations/notebooks/create-raw-data-expectations.py](./great_expectations/notebooks/create-raw-data-expectations.py)
2. Modeling data: [./great_expectations/notebooks/create-modeling-data-expectations.py](./great_expectations/notebooks/create-modeling-data-expectations.py)
3. Holdout error data: [./great_expectations/notebooks/create-holdout-error-data-expectations.py](./great_expectations/notebooks/create-holdout-error-data-expectations.py)

All of the expectation creation scripts follow a similar pattern where we first add the BasicProfiler suite 
as an expectation and then create our expectations as the "default" suite for the DataAsset.
Those default expectations are created by loading the data from the folder as a Batch and 
only need to be done once for the first time or you are updating the expectations.

The choice to use `.py` scripts instead of notebooks is purely for personal preference. 
The scripts are stored in the notebooks folder [`./great_expectations/notebooks`](./great_expectations/notebooks) 
for reference just like a notebook would be.  


---

### Running Your Analysis

In the sections above we described the input data and how we created expectations to validate 
data at three different points of the analysis (before, during, and after modeling). 
There is a script called [`main.py`](main.py) which holds the full end-to-end analysis.
When running the script you should see something like: 

```
$ python main.py

Successfully validated raw data.
Successfully validated modeling data.
Successfully validated holdout error data.
```

In this script there are sections which validate the data against the created expectations. 
At the end of each section there is an `assert` check that the validation run was 
successful. The script will generate an `AssertionError` if any of the expectations are not met. 
If the run was successful you should see the validations of the run stored in 
`uncommitted/validations` of your Great Expectations folder. 


---

### Demo Scenarios

The `main.py` script supports three scenarios to demonstrate how Great Expectations 
can identify changes in your pipeline. The first scenario is an example where the 
raw data is not what you expect because it is missing a column. You can run that 
scenario like this: 

```
$ python main.py missing-column

The following raw data expectations failed:
{'expectation_type': 'expect_table_columns_to_match_ordered_list', 'kwargs': {'column_list': ['species', 'color', 'beak_ratio', 'claw_length', 'wing_density'
, 'weight']}}
{'expectation_type': 'expect_column_values_to_be_of_type', 'kwargs': {'column': 'species', 'type_': 'str'}}
{'expectation_type': 'expect_column_values_to_not_be_null', 'kwargs': {'column': 'species'}}
{'expectation_type': 'expect_column_values_to_be_in_set', 'kwargs': {'column': 'species', 'value_set': ['avis', 'ales']}}
Traceback (most recent call last):
  File "main.py", line 55, in <module>
    assert validation_result_raw_dat["success"]
AssertionError
```

The second scenario is one where a pickled **scikit-learn** transformer is loaded. 
The transformer is supposed to have been created using 2 quantile bins. However, 
this transformer was created using 4 quartile bins so the preprocessed raw data takes 
values [0.0, 1.0, 2.0, 3.0] instead of the expected value set [0.0, 1.0].

```
$ python main.py different-transformer

Successfully validated raw data.
The following modeling data expectations failed:
{'expectation_type': 'expect_column_values_to_be_in_set', 'kwargs': {'column': 'V4', 'value_set': [0.0, 1.0]}}
Traceback (most recent call last):
  File "main.py", line 139, in <module>
    assert validation_result_modeling_dat["success"]
AssertionError
```

In the third scenario we make a small change to the holdout data making one observation 
an extremely large outlier (999.99) which results in a prediction error of more 
than 100 (one of of expectations).

```
$ python main.py holdout-outlier

Successfully validated raw data.
Successfully validated modeling data.
The following holdout error data expectations failed:
{'expectation_type': 'expect_column_values_to_be_between', 'kwargs': {'column': 'error', 'min_value': -100, 'max_value': 100}}
Traceback (most recent call last):
  File "main.py", line 194, in <module>
    assert validation_result_holdout_error_dat["success"]
AssertionError
```


---

[Top](#great-expectations-pipeline-tests-for-scikit-learn)