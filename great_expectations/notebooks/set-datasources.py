#!/usr/bin/env python

import great_expectations as ge


# Load the data context for the project
context = ge.data_context.DataContext()

# Add a new Pandas datasource from the data folder
context.add_datasource(name='data__dir', type_="pandas",
                       base_directory='../data',
                       reader_options={"sep": ",", "header": 0, "engine": "python"})

# Add a new Pandas datasource from the output folder
context.add_datasource(name='output__dir', type_="pandas",
                       base_directory='../output',
                       reader_options={"sep": ",", "header": 0, "engine": "python"})
