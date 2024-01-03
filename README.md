To run the period prediction API in a local environment:

```functions-framework-python --target predict_next_period_start_day```

To deploy the API on Google Cloud Function:

```
gcloud functions deploy period_prediction_api \
    --gen2 \
    --runtime=python311 \
    --region=us-west1 \
    --source=<project_path> \
    --entry-point=predict_next_period_start_day \
    --trigger-http \
    --allow-unauthenticated --memory=8192
```
