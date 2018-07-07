set TRAIN_DATA=gs://deep_impact/ml-input/training_data.csv
set EVAL_DATA=gs://deep_impact/ml-input/eval_data.csv
set STAT_DATA=gs://deep_impact/ml-input/statistics_data.csv
set OUTPUT=gs://deep_impact/ml-output/sample3
cd C:\github\deep_impact\ml\gcp
gcloud ml-engine jobs submit training sample3_v50 --config sample3\config.yaml --region us-east1 --job-dir %OUTPUT% --runtime-version 1.8 --module-name sample3.training --package-path C:\github\deep_impact\ml\gcp\sample3 -- --train-files "%TRAIN_DATA%" --eval-files "%EVAL_DATA%" --statistics-file "%STAT_DATA%" --output %OUTPUT%
rem gcloud ml-engine jobs submit training sample3_v36  --runtime-version 1.1 --region us-east1 --job-dir %OUTPUT% --module-name sample3.training --package-path C:\github\deep_impact\ml\gcp\sample3 -- --train-files "%TRAIN_DATA%" --eval-files "%EVAL_DATA%" --statistics-file "%STAT_DATA%" --output %OUTPUT%
rem gcloud ml-engine jobs submit training sample3_v35  --region us-east1 --job-dir %OUTPUT% --module-name sample3.training --package-path C:\github\deep_impact\ml\gcp\sample3 -- --train-files "%TRAIN_DATA%" --eval-files "%EVAL_DATA%" --statistics-file "%STAT_DATA%" --output %OUTPUT%
rem gcloud ml-engine jobs submit training sample3 --job-dir %OUTPUT% --runtime-version 1.4 --module-name training --package-path ./ --region us-east1 -- --train-files gs://deep_impact/ml-input/training_data.csv --eval-files gs://jrdb/ml-input/eval_data.csv --statistics-file gs://jrdb/ml-input/statistics_data.csv --verbosity DEBUG


