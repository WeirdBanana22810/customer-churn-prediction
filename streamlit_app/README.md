# Customer Churn Prediction System -- Streamlit App

A visual front end for the prediction system built in Phase 6 of the project.
Enter a customer's details in a form, click a button, and see their churn
prediction, probability, risk level, contributing factors, and a recommended
action -- all rendered on screen instead of printed as text in a notebook.

## Files

- `app.py` -- the Streamlit app itself
- `train_and_save_model.py` -- trains the tuned Random Forest (same settings
  found by GridSearchCV in Phase 5) and saves it, the scaler, and the column
  structure to `churn_model.pkl`
- `churn_model.pkl` -- the saved model artifacts (already generated -- you
  don't need to retrain unless you change the data or model settings)
- `customer_churn.csv` -- the dataset, needed only if you re-run
  `train_and_save_model.py`
- `requirements.txt` -- Python packages needed to run this

## Run it locally

1. Open a terminal in this folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Only needed if `churn_model.pkl` isn't already present, or you want to
   retrain): 
   ```
   python train_and_save_model.py
   ```
4. Launch the app:
   ```
   streamlit run app.py
   ```
5. It'll open automatically in your browser at `http://localhost:8501`.

## Deploy it for free (Streamlit Community Cloud)

Since this folder is already inside your GitHub repo:

1. Push this `streamlit_app` folder to your `customer-churn-prediction` repo
   (same `git add . / git commit / git push` routine as always).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click "New app", select your `customer-churn-prediction` repo, set the
   branch to `main`, and set the main file path to `streamlit_app/app.py`.
4. Click "Deploy". You'll get a public link (like
   `yourapp.streamlit.app`) you can share -- same setup you likely already
   used for PlantShield.

## A note on `churn_model.pkl`

This file is about 35 MB. GitHub handles this fine (limit is 100 MB per
file), but it's a binary file, not code -- if you ever retrain the model with
different settings, just re-run `train_and_save_model.py` to regenerate it
and commit the new version.
