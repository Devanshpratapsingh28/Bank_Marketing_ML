import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, static_folder='static', static_url_path='/static')

with open('preprocessing.pkl', 'rb') as f:
    preprocessing_pipeline, feature_names = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


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

            input_data = []

            for col in original_columns:
                val = form_data.get(col)

                if val is None:
                    return render_template(
                        "index.html",
                        error=f"Missing value for {col}"
                    )

                try:
                    num = float(val)

                    if num.is_integer():
                        input_data.append(int(num))
                    else:
                        input_data.append(num)

                except ValueError:
                    input_data.append(val)

            # Convert to DataFrame
            df_data = pd.DataFrame([input_data], columns=original_columns)

            # Apply preprocessing transformation
            pre_data = preprocessing_pipeline.transform(df_data)

           # Make prediction
            prediction = model.predict(pre_data)[0]
            probability = model.predict_proba(pre_data)[0]
            confidence = round(probability[int(prediction)] * 100, 2)

            output = int(prediction)
            print("output", output)

            if prediction == 1:
                result = "The customer will subscribe to the term deposit."
            else:
                result = "The customer will not subscribe to the term deposit."


            return render_template(
                "index.html",
                isSub=result,
                confidence=confidence
            )    


        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)