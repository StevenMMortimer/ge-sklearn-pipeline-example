#!/usr/bin/env python

import great_expectations as ge


# Load the data context for the project
context = ge.data_context.DataContext()

# Profile the new data asset ---------------------------------------------------
context.profile_datasource(datasource_name='output__dir', data_assets=['modeling-data'])
context.build_data_documentation()

# Create expectations ----------------------------------------------------------
df = context.get_batch('modeling-data')

# Expect that the dataset names are V[0-9]+ because sklearn modeling datasets
# are arrays without column names. We just assigned these names when writing
# out the dataset.
expected_colnames = ['V' + str(x) for x in range(6)]
df.expect_table_columns_to_match_ordered_list(expected_colnames)

# Expect that all columns are numeric datatypes because many sklearn estimators
# require that the training data has been encoded as all numerics
ge.dataset.util.create_multiple_expectations(df, expected_colnames,
                                             'expect_column_values_to_be_of_type',
                                             type_='float64')

# Expect that all columns are non-null, again, because many sklearn estimators
# will not work with missing values or may not work as desired by default
ge.dataset.util.create_multiple_expectations(df, expected_colnames,
                                             'expect_column_values_to_not_be_null')

# Expect that the dummy coded variables only come from the set of 0 or 1
df.expect_column_values_to_be_in_set(column='V0', value_set=[0.0, 1.0])
df.expect_column_values_to_be_in_set(column='V1', value_set=[0.0, 1.0])

# Expect that the ordinal coded variables only come from the set of 0, 1, 2
# which represent the three different levels ['low', 'moderate', 'high']
df.expect_column_values_to_be_in_set(column='V2', value_set=[0.0, 1.0, 2.0])

# Expect that the variable cut across [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] is only one of five values
df.expect_column_values_to_be_in_set(column='V3', value_set=[0.0, 1.0, 2.0, 3.0, 4.0])

# Expect that the binned variable using only 2 quantiles come from the set of 0 or 1
# NOTE: This is a helpful expectation to catch an event like another analyst changing
# the binning strategy to quartiles (4 bins).
df.expect_column_values_to_be_in_set(column='V4', value_set=[0.0, 1.0])

# Expect that the target variable (weight) falls within an expected range which
# may prevent us from modeling outliers or incorrectly coded or transformed data
df.expect_column_values_to_be_between(column='V5', min_value=100, max_value=250)

# Save expectations ------------------------------------------------------------
df.save_expectation_suite()
