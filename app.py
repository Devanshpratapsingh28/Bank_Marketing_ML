import gzip
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load preprocessing pipeline and expected feature names
with open('preprocessing.pkl', 'rb') as f:
    preprocessing_pipeline, feature_names = pickle.load(f)

# Load trained RandomForestClassifier model
with gzip.open("rf.pkl.gz", "rb") as f:
    model = pickle.load(f)

# Define original input feature names (before preprocessing)
original_columns = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'duration', 'campaign',
    'previous', 'poutcome'
]

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")

    elif request.method == "POST":
        try:
            form_data = request.form.to_dict()

            # Convert input values to the correct format (numeric/categorical)
            input_data = []
            for col in original_columns:
                val = form_data.get(col)
                if val is None:
                    return render_template("index.html", error=f"Missing value for {col}")

                # Convert to float if possible
                try:
                    float_val = float(val)
                    input_data.append(int(float_val) if float_val.is_integer() else round(float_val, 2))
                except ValueError:
                    input_data.append(val)  # Keep categorical values as strings

            # Convert to DataFrame
            df_data = pd.DataFrame([input_data], columns=original_columns)

            # Apply preprocessing transformation
            pre_data = preprocessing_pipeline.transform(df_data)

            # Make prediction
            prediction = model.predict(pre_data)
            output = int(prediction[0])
            print(output)
            result = ""
            if output == 1:
                result =  "The customer will subscribe to the term deposit."
            else:
                result = "The customer will not subscribe to the term deposit."
            return render_template("index.html",isSub=result)    


        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)