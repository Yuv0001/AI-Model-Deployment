FROM python:3.10

WORKDIR  /AI-Role-assignment

COPY . .

RUN ls -l && \
    pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000

ENV DATABASE_URL "mysql+pymysql://root:Yuv123@localhost:3306/classifiermodel"

CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
 
