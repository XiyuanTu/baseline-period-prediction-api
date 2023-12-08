from flask import Flask
from flask import request

app = Flask(__name__)

import sys

sys.path.append('./prediction/scripts')
from model_predict import *

@app.route('/predict', methods=['GET'])
def predict_next_period_start_day():  # put application's code here

    history = request.args.get("history")
    data = [int(length) for length in history.split(" ")]
    # print(data)
    result = predict(data)
    # result = predict([27, 25, 30, 26, 28, 30, 25, 27, 29, 29])

    return {
        "result": result
    }

if __name__ == '__main__':
    load_model("./prediction/scripts/model.pt")
    app.run()
