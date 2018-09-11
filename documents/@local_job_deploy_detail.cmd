cmd /c bq rm -t jrdb_raw_data.%1
cmd /c bq mk --time_partitioning_field partitioning_date --table jrdb_raw_data.%1 assets\bigquery\definition_data\ddl\jrdb_raw_data\%1.json
timeout 10