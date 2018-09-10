cmd /c gsutil -o GSUtil:default_project_id=yu-it-base cp assets\bigquery\definition_data\csv\data_characteristics.csv gs://deep_impact/assets/bigquery/definition_data/csv/
cmd /c bq rm -t -f jrdb_raw_data_schema_info.data_characteristics
cmd /c bq load  --skip_leading_rows=1 jrdb_raw_data_schema_info.data_characteristics gs://deep_impact/assets/bigquery/definition_data/csv/data_characteristics.csv assets\bigquery\definition_data\schema_info_ddl\data_characteristics.json
timeout 10
