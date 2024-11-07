from flask import Flask, request, render_template, jsonify
import pandas as pd
from models.forecast_models import seasonal_naive_forecast, holt_winters_forecast, sarima_forecast, linear_regression_forecast
import os

app = Flask(__name__)
app.config.from_object('config.Config')

# Route for uploading CSV file and displaying the predictions
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and process the forecast
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Read CSV
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df[df['store'] == 1]
    df = df[df['item'] == 1]

    # Split the data
    df = df.set_index('date')
    train_df = df.loc[:'2017-09-30']
    test_df = df.loc['2017-10-01':]

    # Get predictions from different models
    results = {
        'Seasonal_Naive': seasonal_naive_forecast(train_df, test_df),
        'Holt_Winters': holt_winters_forecast(train_df, test_df),
        'SARIMA': sarima_forecast(train_df, test_df),
        'Linear_Regression': linear_regression_forecast(train_df, test_df)
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
