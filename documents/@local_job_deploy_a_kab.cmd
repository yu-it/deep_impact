cmd /c bq rm -t jrdb_raw_data.a_kab
cmd /c bq mk --table jrdb_raw_data.a_kab assets\bigquery\definition_data\ddl\jrdb_raw_data\a_kab.json
timeout 10