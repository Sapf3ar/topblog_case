FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r "requirements.txt"
RUN apt-get update && apt-get install libgl1  -y

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]