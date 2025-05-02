
# Kidney Disease Prediction API

A machine learning model to predict chronic kidney disease, deployed with Flask.

## How to Run Locally

1. Install dependencies:

pip install -r requirements.txt

2. Train the model:

python kidney_model.py

3. Run the Flask app:

python app.py

4. Test API:
Use Postman or curl:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":48, "bp":80, "sg":1.02, ...}'
