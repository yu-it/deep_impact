cmd /c gsutil -o GSUtil:default_project_id=yu-it-base cp assets\bigquery\definition_data\csv\linear_interpolation_params.csv gs://deep_impact/assets/bigquery/definition_data/csv/
cmd /c bq rm -t -f deep_impact_vector_data_control_info.linear_interpolation_params
cmd /c bq load  --skip_leading_rows=1 deep_impact_vector_data_control_info.linear_interpolation_params gs://deep_impact/assets/bigquery/definition_data/csv/linear_interpolation_params.csv assets\bigquery\definition_data\ddl\deep_impact_vector_data_control_info\linear_interpolation_params.json
timeout 10
