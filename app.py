from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
model_path = os.path.join("artifacts", "model.pkl")

preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)

cat_pipeline = preprocessor.named_transformers_['cat_pipeline']
one_hot = cat_pipeline.named_steps['one_hot_encoder']

expected_categories = {}
for i, col in enumerate(cat_pipeline.feature_names_in_):
    expected_categories[col] = one_hot.categories_[i]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method=='POST':
        try:
            data = {
                'gender': [request.form.get('gender')],
                'race/ethnicity': [request.form.get('ethnicity')],
                'parental level of education': [request.form.get('parental_level_of_education')],
                'lunch': [request.form.get('lunch')],
                'test preparation course': [request.form.get('test_preparation_course')],
                'reading score': [float(request.form.get('reading_score'))],
                'writing score': [float(request.form.get('writing_score'))]
            }

            df = pd.DataFrame(data)

            for col in expected_categories:
                df[col] = df[col].apply(lambda x: x if x in expected_categories[col] else expected_categories[col][0])

            X_transformed = preprocessor.transform(df)
            prediction = model.predict(X_transformed)[0]
            # print(prediction)
            app.logger.info(f"Prediction: {prediction}")
            return render_template('home.html', result=round(prediction, 2))

        except Exception as e:
            return render_template('home.html', error=str(e))
    else:
        return render_template('home.html',error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
