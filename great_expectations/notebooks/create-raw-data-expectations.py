#!/usr/bin/env python

import great_expectations as ge

from lib.constants import *


# Load the data context for the project
context = ge.data_context.DataContext()

# Profile the new data asset ---------------------------------------------------
context.profile_datasource(datasource_name='data__dir', data_assets=['raw-data'])
context.build_data_documentation()

# Create expectations ----------------------------------------------------------
df = context.get_batch('raw-data')

# Expect that the raw data contains the same columns
# NOTE: This can be pulled directly from the data using `df.get_table_columns()`
# if you believe the initial parsing is correct correct and shouldn't change.
expected_colnames = ['species', 'color', 'beak_ratio', 'claw_length', 'wing_density', 'weight']
df.expect_table_columns_to_match_ordered_list(expected_colnames)

# Expect the raw data is loaded with the following data types
# NOTE: This can also be pulled directly from the data using `df.dtypes` if you believe
# the initial parsing is correct correct and shouldn't change.
df.expect_column_values_to_be_of_type(column='species', type_='str')
df.expect_column_values_to_be_of_type(column='color', type_='str')
df.expect_column_values_to_be_of_type(column='beak_ratio', type_='str')
df.expect_column_values_to_be_of_type(column='claw_length', type_='float64')
df.expect_column_values_to_be_of_type(column='wing_density', type_='float64')
df.expect_column_values_to_be_of_type(column='weight', type_='float64')

# Expect that the number of existing values and values are not null in each column
df.expect_table_row_count_to_equal(16)
ge.dataset.util.create_multiple_expectations(df, expected_colnames,
                                             'expect_column_values_to_not_be_null')

# Expect that the string column values only come from a set list of known values
df.expect_column_values_to_be_in_set(column='species', value_set=SPECIES)
df.expect_column_values_to_be_in_set(column='color', value_set=COLOR)
df.expect_column_values_to_be_in_set(column='beak_ratio', value_set=BEAK_RATIO)

# Save expectations ------------------------------------------------------------
df.save_expectation_suite()
