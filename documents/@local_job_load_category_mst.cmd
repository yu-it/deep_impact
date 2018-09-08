cmd /c bq rm -t -f jrdb_raw_data_schema_info.category_mst
cmd /c gsutil -o GSUtil:default_project_id=yu-it-base cp assets\bigquery\definition_data\csv\category_mst.csv gs://deep_impact/assets/bigquery/definition_data/csv/
cmd /c bq load  --skip_leading_rows=1 jrdb_raw_data_schema_info.category_mst gs://deep_impact/assets/bigquery/definition_data/csv/category_mst.csv assets\bigquery\definition_data\schema_info_ddl\category_mst.json
timeout 10
