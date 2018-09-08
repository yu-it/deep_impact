cmd /c bq rm -t jrdb_raw_data.a_kyi
cmd /c bq mk --table jrdb_raw_data.a_kyi assets\bigquery\definition_data\ddl\jrdb_raw_data\a_kyi.json
timeout 10