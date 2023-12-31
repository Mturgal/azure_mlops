#!/bin/bash

run_id=$(az ml job create --file src/production_job.yml --resource-group Learn_MLOps --workspace-name mlops_ws_prod --query name -o tsv)
if [[ -z "$run_id" ]]
then
  echo "Job creation failed"
  exit 3
fi
az ml job show -n $run_id --resource-group Learn_MLOps --workspace-name mlops_ws_prod --web
status=$(az ml job show -n $run_id   --query status -o tsv --resource-group Learn_MLOps --workspace-name mlops_ws_prod)
if [[ -z "$status" ]]
then
  echo "Status query failed and run id: $run_id"
  exit 4
fi
running=("NotStarted" "Queued" "Starting" "Preparing" "Running" "Finalizing" "CancelRequested")
while [[ ${running[*]} =~ $status ]]
do
  sleep 15 
  status=$(az ml job show -n $run_id  --query status -o tsv --resource-group Learn_MLOps --workspace-name mlops_ws_prod)
  echo "$status and run id: $run_id"

done
if [[ "$status" != "Completed" ]]  
then
  echo "Training Job failed or canceled"
  exit 3
fi
echo "run_id=$run_id" >> $GITHUB_OUTPUT


