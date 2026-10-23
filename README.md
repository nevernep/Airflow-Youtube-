# Airflow YouTube Views Prediction

โปรเจกต์ตัวอย่าง Machine Learning pipeline สำหรับทำนายจำนวน Views ของวิดีโอ
YouTube โดยใช้ Apache Airflow, Random Forest และ FastAPI

## ภาพรวม

โปรเจกต์ประกอบด้วยส่วนหลักดังนี้

- **Apache Airflow**: orchestration สำหรับเตรียมข้อมูล, train, evaluate และ deploy โมเดล
- **PostgreSQL**: ฐานข้อมูล metadata ของ Airflow และฐานข้อมูลปลายทางสำหรับเก็บ metric
- **YouTube Model API**: FastAPI สำหรับเรียกใช้โมเดลที่ deploy ล่าสุด
- **Champion-Challenger**: เปรียบเทียบโมเดลใหม่กับโมเดลปัจจุบันก่อน deploy

โมเดลใช้ `RandomForestRegressor` และ feature ต่อไปนี้:

`category_id`, `likes`, `dislikes`, `comment_count`, `comments_disabled`,
`ratings_disabled`, `publish_hour`, `publish_day_of_week`, `title_length`,
`tag_count`

## โครงสร้างสำคัญ

```text
.
├── dags/
│   ├── ml_01_youtube_pipeline.py       # Train และ deploy ตาม R² threshold
│   └── ml_02_youtube_pipeline.py       # Champion-Challenger pipeline
├── model_service/
│   ├── main.py                         # FastAPI application และ web UI
│   ├── youtube.py                      # Prediction router
│   ├── Dockerfile
│   └── requirements.txt
├── data/youtube/
│   └── USvideos.csv                   # Dataset จาก Kaggle (ไม่เก็บใน Git)
├── models/youtube_models/
│   └── current_model.pkl               # โมเดลที่ deploy ล่าสุด (ไม่เก็บใน Git)
├── docker-compose.yaml
└── .env
```

## สิ่งที่ต้องมี

- Docker Desktop พร้อม Docker Compose
- พื้นที่ดิสก์เพียงพอสำหรับ Docker images, dataset และโมเดล
- Dataset `USvideos.csv` จาก [Kaggle Trending YouTube Video Statistics](https://www.kaggle.com/datasets/datasnaek/youtube-new)

วางไฟล์ dataset ที่:

```text
data/youtube/USvideos.csv
```

ไฟล์ dataset และโมเดล `.pkl` ถูกระบุใน `.gitignore` เนื่องจากมีขนาดใหญ่เกินข้อจำกัดของ GitHub

## การติดตั้งและเริ่มระบบ

เปิด terminal ที่โฟลเดอร์โปรเจกต์ แล้วรัน:

```bash
docker compose up airflow-init
docker compose up -d
```

ตรวจสอบ container:

```bash
docker compose ps
```

เมื่อระบบพร้อมใช้งาน:

- Airflow: <http://localhost:8080>
- Model API: <http://localhost:8001>
- Swagger UI: <http://localhost:8001/docs>
- API health check: <http://localhost:8001/health>

ค่าเริ่มต้นของ Airflow จาก `docker-compose.yaml`:

```text
Username: airflow
Password: airflow
```

ควรเปลี่ยน password ก่อนใช้งานจริง

## ตั้งค่า Airflow Connection

DAG `youtube_champion_pipeline_dag` ใช้ Connection ID ชื่อ `postgres_target`

ตั้งค่าผ่าน Airflow UI ที่ **Admin > Connections**:

| Field | Value |
|---|---|
| Connection Id | `postgres_target` |
| Connection Type | `Postgres` |
| Host | `postgres_target` |
| Port | `5432` |
| Database | `etl_db` |
| Login | `etluser` |
| Password | `etlpass` |

หรือใช้ CLI ภายใน container:

```bash
docker compose run --rm airflow-cli \
  connections add postgres_target \
  --conn-type postgres \
  --conn-host postgres_target \
  --conn-port 5432 \
  --conn-schema etl_db \
  --conn-login etluser \
  --conn-password etlpass
```

## การใช้งาน DAG

ทั้งสอง DAG ตั้งค่า `schedule=None` จึงต้อง trigger ด้วยตนเองจาก Airflow UI หรือ CLI

### 1. Basic pipeline

`youtube_views_pipeline_dag`

ขั้นตอนหลัก:

1. อ่านและเตรียมข้อมูล `USvideos.csv`
2. สร้าง feature
3. Train `RandomForestRegressor`
4. ประเมินด้วย RMSE และ R²
5. Deploy เมื่อ R² ผ่านเกณฑ์ `0.70`
6. ทำ smoke test หลัง deploy

### 2. Champion-Challenger pipeline

`youtube_champion_pipeline_dag`

ขั้นตอนหลัก:

1. สร้าง challenger model จากข้อมูลล่าสุด
2. ประเมิน challenger ด้วย RMSE และ R²
3. อ่าน metric ของ champion จาก PostgreSQL
4. Deploy challenger เฉพาะเมื่อผลลัพธ์ดีกว่า champion
5. บันทึก metric ลงตาราง `youtube_model_metrics`
6. ทำ smoke test หลัง deploy

Trigger ผ่าน CLI:

```bash
docker compose run --rm airflow-cli \
  dags trigger youtube_views_pipeline_dag

docker compose run --rm airflow-cli \
  dags trigger youtube_champion_pipeline_dag
```

โมเดลที่ deploy สำเร็จจะถูกเขียนไว้ที่:

```text
models/youtube_models/current_model.pkl
```

## Model API

### Health check

```bash
curl http://localhost:8001/health
```

### ตรวจสอบข้อมูลโมเดล

```bash
curl http://localhost:8001/youtube/model-info
```

### ทำนายจำนวน Views

```bash
curl -X POST http://localhost:8001/youtube/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"category_id\":24,\"likes\":25000,\"dislikes\":350,\"comment_count\":1800,\"comments_disabled\":false,\"ratings_disabled\":false,\"publish_hour\":18,\"publish_day_of_week\":5,\"title_length\":58,\"tag_count\":12}"
```

ตัวอย่าง response:

```json
{
  "predicted_views": 123456,
  "model_name": "youtube_views_rf",
  "model_last_modified": "2026-01-01T12:00:00"
}
```

นอกจาก API แล้ว ยังเปิดหน้า web UI ได้ที่ <http://localhost:8001>

## การดู log และการหยุดระบบ

ดู log ของทุก service:

```bash
docker compose logs -f
```

ดู log เฉพาะ Model API:

```bash
docker compose logs -f model_api
```

หยุดระบบโดยไม่ลบ volume:

```bash
docker compose down
```

หากต้องการลบ volume ของ PostgreSQL และข้อมูลที่เก็บใน Docker ด้วย ให้ใช้คำสั่งนี้ด้วยความระมัดระวัง:

```bash
docker compose down -v
```

## หมายเหตุด้านการพัฒนา

- Airflow containers mount โฟลเดอร์ `dags`, `data`, `models` และ `logs` จากเครื่อง host
- Model API mount `models` แบบ read-only และจะโหลด `current_model.pkl` ใหม่ทุกครั้งที่มี request
- หากยังไม่มี `current_model.pkl`, endpoint prediction จะตอบสถานะ `503` จนกว่าจะรัน task `deploy_model` สำเร็จ
- ข้อมูลรหัสผ่านใน `docker-compose.yaml` เป็นค่าเริ่มต้นสำหรับ local development เท่านั้น

## Repository

GitHub: <https://github.com/nevernep/Airflow-Youtube->
