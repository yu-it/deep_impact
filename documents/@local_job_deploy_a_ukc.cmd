cmd /c bq rm -t jrdb_raw_data.a_ukc
cmd /c bq mk --table jrdb_raw_data.a_ukc assets\bigquery\definition_data\ddl\jrdb_raw_data\a_ukc.json
timeout 10