cd C:\github\deep_impact\ml\gcp\sample3
set MODEL_NAME=iris
set INPUT_DATA_FILE=C:\github\deep_impact\ml\gcp\sample3\input.csv
set VERSION_NAME=iris_v3
rem gcloud ml-engine predict ���g�p���āA�C���X�^���X���f�v���C�ς݃��f���ɑ��M���܂��B--version �͔C�ӂł��邱�Ƃɒ��ӂ��Ă��������B

rem gcloud ml-engine predict  --runtime-version 1.4 --model %MODEL_NAME% --version %VERSION_NAME% --text-instances %INPUT_DATA_FILE%
gcloud ml-engine predict --model %MODEL_NAME% --version %VERSION_NAME% --text-instances %INPUT_DATA_FILE%
