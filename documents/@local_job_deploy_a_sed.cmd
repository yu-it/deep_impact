cmd /c bq rm -t jrdb_raw_data.a_sed
cmd /c bq mk --table jrdb_raw_data.a_sed assets\bigquery\definition_data\ddl\jrdb_raw_data\a_sed.json
timeout 10