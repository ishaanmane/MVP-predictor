from flask import Flask, render_template, request
import pandas as pd
import pickle
from ml_model import predict_it, add_ranks, stats, model, predictors

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        year = int(request.form['year'])
        result = predict_it(stats, model, year, predictors)
        result_df = pd.concat(result)
        result_html = result_df.to_html(classes='table table-striped')
        return render_template('result.html', result_html=result_html)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)