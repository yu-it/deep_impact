cmd /c bq rm -t jrdb_raw_data.a_kza
cmd /c bq mk --table jrdb_raw_data.a_kza assets\bigquery\definition_data\ddl\jrdb_raw_data\a_kza.json
timeout 10