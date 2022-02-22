FROM python

RUN mkdir -p /app

COPY requirements.txt . 
#copy dependencies into requirements txt

RUN pip install -r requirements.txt
#install dependencies

COPY . /app

EXPOSE 8000

CMD ["python", "/app/app.py"]