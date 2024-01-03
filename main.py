from flask import Flask
import functions_framework
import sys

app = Flask(__name__)

sys.path.append('prediction')
from model_predict import *

@functions_framework.http
def predict_next_period_start_day(request):
    load_model("./prediction/model.pt")
    history = request.args.get("history")
    data = [int(length) for length in history.split(" ")]
    result = predict(data)
    return {
        "result": result
    }
