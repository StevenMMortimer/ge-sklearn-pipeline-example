#!/usr/bin/env python

import great_expectations as ge

# Load the data context for the project
context = ge.data_context.DataContext()

# Profile the new data asset ---------------------------------------------------
context.profile_datasource(datasource_name='output__dir', data_assets=['holdout-error-data'])
context.build_data_documentation()

# Create expectations ----------------------------------------------------------
df = context.get_batch('holdout-error-data')

# Expect that predictions are off by no more than 100
df.expect_column_values_to_be_between(column='error', min_value=-100, max_value=100)

# Expect that roughly 95% of the error values fall within -20 and 20.
df.expect_column_kl_divergence_to_be_less_than(column='error',
                                               partition_object={'bins': [-20, 0, 20],
                                                                 'weights': [0.475, 0.475],
                                                                 'tail_weights': [0.025, 0.025]},
                                               threshold=0.5)

# Save expectations ------------------------------------------------------------
df.save_expectation_suite()
