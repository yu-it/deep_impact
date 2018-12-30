call "@local_job_deploy_a_%1.cmd"
gcloud beta dataflow jobs run load_%1 --gcs-location gs://deep_impact/dataflow/jrdb_%1_loader
