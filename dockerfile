FROM python:3.10.6-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY predicting_social_anxiety predicting_social_anxiety
COPY cb2_model.joblib cb2_model.joblib
COPY cb2_preprocessor.joblib cb2_preprocessor.joblib
CMD uvicorn predicting_social_anxiety.api.modelapi:app --host 0.0.0.0 --port $PORT
