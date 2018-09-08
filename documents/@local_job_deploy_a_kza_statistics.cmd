cmd /c bq rm -t jrdb_raw_data_schema_info.a_kza_statistics
cmd /c bq mk --table jrdb_raw_data_schema_info.a_kza_statistics assets\bigquery\definition_data\schema_info_ddl\statistics.json
timeout 10