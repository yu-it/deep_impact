set TRAIN_DATA=C:\github\deep_impact\ml\gcp\sample3\local_data\training_data.csv
set EVAL_DATA=C:\github\deep_impact\ml\gcp\sample3\local_data\eval_data.csv
set OUTPUT=C:\github\deep_impact\ml\gcp\sample3\local_output_gcloud_test

cd C:\github\deep_impact\
gcloud ml-engine local train  --package-path C:\github\deep_impact\ml --module-name ml.gcp.sample3.training  -- --train-files "%TRAIN_DATA%" --eval-files "%EVAL_DATA%" --output %OUTPUT%

