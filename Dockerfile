FROM python:3.10-slim

WORKDIR /app

COPY ["requirements.txt" , "./"]

RUN pip install -r requirements.txt

COPY ["predict.py", "model_xgb.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]