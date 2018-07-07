cd C:\github\deep_impact\ml\gcp\sample3
set MODEL_NAME=iris
set INPUT_DATA_FILE=C:\github\deep_impact\ml\gcp\sample3\input.csv
set VERSION_NAME=iris_v3
rem gcloud ml-engine predict を使用して、インスタンスをデプロイ済みモデルに送信します。--version は任意であることに注意してください。

rem gcloud ml-engine predict  --runtime-version 1.4 --model %MODEL_NAME% --version %VERSION_NAME% --text-instances %INPUT_DATA_FILE%
gcloud ml-engine predict --model %MODEL_NAME% --version %VERSION_NAME% --text-instances %INPUT_DATA_FILE%
