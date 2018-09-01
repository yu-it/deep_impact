cmd /c bq rm -t jrdb_raw_data.a_bac
cmd /c bq mk --table jrdb_raw_data.a_bac assets\bigquery\definition_data\ddl\a_bac.json
timeout 10