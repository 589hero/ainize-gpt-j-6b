FROM 589hero/gpt-j-6b-fp16:v1.0.0

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app

COPY . .

EXPOSE 5000

CMD ["python3", "app.py"]