# Dockerfile — YouTube Views Model Serving API
# Workshop 2
# แยก image ออกจาก Airflow โดยสิ้นเชิง

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# เปลี่ยนจาก weather.py -> youtube.py
COPY ["main.py", "youtube.py", "./"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]