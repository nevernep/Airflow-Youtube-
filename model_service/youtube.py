"""
youtube.py — YouTube Views Model Router
=======================================
Workshop 2: YouTube Views Prediction

โมดูล FastAPI สำหรับเสิร์ฟโมเดล YouTube Views Prediction
ที่ Apache Airflow deploy ไว้

โมเดลมาจาก DAG:
- youtube_views_pipeline_dag
- youtube_champion_pipeline_dag

Airflow เขียนโมเดลไว้ที่:
    /opt/airflow/models/youtube_models/current_model.pkl

ส่วน model_api container mount ./models ไว้ที่:
    /models

ดังนั้นไฟล์นี้อ่านโมเดลจาก:
    /models/youtube_models/current_model.pkl

ทุกครั้งที่มี request จะโหลดโมเดลจาก disk ใหม่
เพื่อให้ใช้โมเดลล่าสุดทันทีหลัง Airflow deploy
โดยไม่ต้อง restart service
"""

import os
from datetime import datetime

import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


# =========================================================
# Configuration
# =========================================================

YOUTUBE_MODEL_PATH = (
    "/models/youtube_models/current_model.pkl"
)

MODEL_NAME = "youtube_views_rf"


FEATURE_COLUMNS = [
    "category_id",
    "likes",
    "dislikes",
    "comment_count",
    "comments_disabled",
    "ratings_disabled",
    "publish_hour",
    "publish_day_of_week",
    "title_length",
    "tag_count",
]


router = APIRouter(
    prefix="/youtube",
    tags=["YouTube Views Prediction"],
)


# =========================================================
# Request / Response
# =========================================================

class PredictYouTubeRequest(BaseModel):

    category_id: int = Field(
        ...,
        ge=0,
        example=24,
    )

    likes: int = Field(
        ...,
        ge=0,
        example=25000,
    )

    dislikes: int = Field(
        ...,
        ge=0,
        example=350,
    )

    comment_count: int = Field(
        ...,
        ge=0,
        example=1800,
    )

    comments_disabled: bool = Field(
        False,
        example=False,
    )

    ratings_disabled: bool = Field(
        False,
        example=False,
    )

    publish_hour: int = Field(
        ...,
        ge=0,
        le=23,
        example=18,
    )

    publish_day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        example=5,
        description="0=Monday ... 6=Sunday",
    )

    title_length: int = Field(
        ...,
        ge=0,
        example=58,
    )

    tag_count: int = Field(
        ...,
        ge=0,
        example=12,
    )


class PredictYouTubeResponse(BaseModel):

    predicted_views: int

    model_name: str

    model_last_modified: str


# =========================================================
# Model Utilities
# =========================================================

def load_youtube_model():
    """
    โหลด current_model.pkl ทุก request
    """

    if not os.path.exists(
        YOUTUBE_MODEL_PATH
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "ยังไม่มี YouTube model "
                "current_model.pkl — "
                "กรุณารัน DAG "
                "youtube_views_pipeline_dag "
                "หรือ youtube_champion_pipeline_dag "
                "จน task deploy_model สำเร็จก่อน"
            ),
        )

    try:

        return joblib.load(
            YOUTUBE_MODEL_PATH
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "โหลด YouTube model "
                f"ไม่สำเร็จ: {exc}"
            ),
        )


def youtube_model_file_info():
    """
    ใช้โดย /health ใน main.py
    """

    exists = os.path.exists(
        YOUTUBE_MODEL_PATH
    )

    last_modified = None

    if exists:

        last_modified = (
            datetime.fromtimestamp(
                os.path.getmtime(
                    YOUTUBE_MODEL_PATH
                )
            ).isoformat()
        )

    return {
        "exists": exists,
        "path": YOUTUBE_MODEL_PATH,
        "last_modified": last_modified,
    }


# =========================================================
# Predict Endpoint
# =========================================================

@router.post(
    "/predict",
    response_model=PredictYouTubeResponse,
)
def predict_youtube(
    payload: PredictYouTubeRequest,
):

    model = load_youtube_model()

    # ต้องเรียง Feature
    # ให้ตรงกับตอน Train 100%
    features = [[
        payload.category_id,
        payload.likes,
        payload.dislikes,
        payload.comment_count,
        int(
            payload.comments_disabled
        ),
        int(
            payload.ratings_disabled
        ),
        payload.publish_hour,
        payload.publish_day_of_week,
        payload.title_length,
        payload.tag_count,
    ]]

    try:

        prediction = model.predict(
            features
        )[0]

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "โมเดล Predict ไม่สำเร็จ: "
                f"{exc}"
            ),
        )

    predicted_views = int(
        round(
            max(
                0,
                float(prediction),
            )
        )
    )

    last_modified = (
        datetime.fromtimestamp(
            os.path.getmtime(
                YOUTUBE_MODEL_PATH
            )
        ).isoformat()
    )

    return PredictYouTubeResponse(
        predicted_views=predicted_views,
        model_name=MODEL_NAME,
        model_last_modified=last_modified,
    )


# =========================================================
# Model Info Endpoint
# =========================================================

@router.get("/model-info")
def youtube_model_info():

    return {
        "model_name": MODEL_NAME,
        "features": FEATURE_COLUMNS,
        "model": youtube_model_file_info(),
    }


# =========================================================
# HTML Fragment
# =========================================================

YOUTUBE_FORM_HTML = """
<hr style="
    margin:32px 0;
    border:none;
    border-top:1px solid #e5e7eb;
">

<h1>
▶ YouTube Views Predictor
</h1>

<p style="
    color:#6b7280;
    font-size:14px;
">

กรอกข้อมูลของ YouTube Video
แล้วกด Predict เพื่อให้โมเดลล่าสุด
ที่ Airflow deploy ไว้คาดการณ์จำนวน Views

</p>


<div class="field">

<label for="category_id">
Video Category
</label>

<select id="category_id">

<option value="1">
Film & Animation
</option>

<option value="2">
Autos & Vehicles
</option>

<option value="10">
Music
</option>

<option value="15">
Pets & Animals
</option>

<option value="17">
Sports
</option>

<option value="19">
Travel & Events
</option>

<option value="20">
Gaming
</option>

<option value="22">
People & Blogs
</option>

<option value="23">
Comedy
</option>

<option
    value="24"
    selected
>
Entertainment
</option>

<option value="25">
News & Politics
</option>

<option value="26">
Howto & Style
</option>

<option value="27">
Education
</option>

<option value="28">
Science & Technology
</option>

<option value="29">
Nonprofits & Activism
</option>

<option value="43">
Shows
</option>

</select>

<div style="
    margin-top:6px;
    color:#777;
    font-size:11px;
    line-height:1.5;
">

    Category from the YouTube Kaggle dataset.
    The selected category is automatically converted
    to its Category ID before prediction.

</div>

</div>


<div class="field">

<label>
Likes
</label>

<input
    type="number"
    id="likes"
    value="25000"
    min="0"
>

</div>


<div class="field">

<label>
Dislikes
</label>

<input
    type="number"
    id="dislikes"
    value="350"
    min="0"
>

</div>


<div class="field">

<label>
Comment Count
</label>

<input
    type="number"
    id="comment_count"
    value="1800"
    min="0"
>

</div>


<div class="field">

<label>
Publish Hour
</label>

<input
    type="number"
    id="publish_hour"
    value="18"
    min="0"
    max="23"
>

</div>


<div class="field">

<label>
Publish Day of Week
</label>

<select id="publish_day_of_week">

<option value="0">
Monday
</option>

<option value="1">
Tuesday
</option>

<option value="2">
Wednesday
</option>

<option value="3">
Thursday
</option>

<option value="4">
Friday
</option>

<option
    value="5"
    selected
>
Saturday
</option>

<option value="6">
Sunday
</option>

</select>

</div>


<div class="field">

<label>
Title Length
</label>

<input
    type="number"
    id="title_length"
    value="58"
    min="0"
>

</div>


<div class="field">

<label>
Tag Count
</label>

<input
    type="number"
    id="tag_count"
    value="12"
    min="0"
>

</div>


<div class="field">

<label>

<input
    type="checkbox"
    id="comments_disabled"
    style="width:auto;"
>

Comments Disabled

</label>

</div>


<div class="field">

<label>

<input
    type="checkbox"
    id="ratings_disabled"
    style="width:auto;"
>

Ratings Disabled

</label>

</div>


<button onclick="doPredictYouTube()">

Predict Views

</button>


<div id="youtube_result"></div>


<script>

async function doPredictYouTube() {

    const payload = {

        category_id:
            parseInt(
                document
                .getElementById(
                    'category_id'
                )
                .value
            ),

        likes:
            parseInt(
                document
                .getElementById(
                    'likes'
                )
                .value
            ),

        dislikes:
            parseInt(
                document
                .getElementById(
                    'dislikes'
                )
                .value
            ),

        comment_count:
            parseInt(
                document
                .getElementById(
                    'comment_count'
                )
                .value
            ),

        comments_disabled:
            document
            .getElementById(
                'comments_disabled'
            )
            .checked,

        ratings_disabled:
            document
            .getElementById(
                'ratings_disabled'
            )
            .checked,

        publish_hour:
            parseInt(
                document
                .getElementById(
                    'publish_hour'
                )
                .value
            ),

        publish_day_of_week:
            parseInt(
                document
                .getElementById(
                    'publish_day_of_week'
                )
                .value
            ),

        title_length:
            parseInt(
                document
                .getElementById(
                    'title_length'
                )
                .value
            ),

        tag_count:
            parseInt(
                document
                .getElementById(
                    'tag_count'
                )
                .value
            ),
    };


    const resultDiv =
        document.getElementById(
            'youtube_result'
        );


    resultDiv.style.display =
        'block';

    resultDiv.className =
        '';

    resultDiv.innerHTML =
        'กำลังคาดการณ์...';


    try {

        const res =
            await fetch(
                '/youtube/predict',
                {
                    method: 'POST',

                    headers: {
                        'Content-Type':
                            'application/json'
                    },

                    body:
                        JSON.stringify(
                            payload
                        ),
                }
            );


        const data =
            await res.json();


        if (!res.ok) {

            resultDiv.className =
                'err';

            resultDiv.innerHTML =
                '❌ '
                + (
                    data.detail
                    ||
                    'เกิดข้อผิดพลาด'
                );

            return;
        }


        const views =
            Number(
                data.predicted_views
            ).toLocaleString();


        resultDiv.className =
            'ok';


        resultDiv.innerHTML = `

            <div style="
                font-size:18px;
                font-weight:bold;
            ">

                Estimated Views:
                ${views}

            </div>

            <div class="meta">
                Model:
                ${data.model_name}
            </div>

            <div class="meta">
                โมเดลอัปเดตล่าสุด:
                ${data.model_last_modified}
            </div>
        `;

    }

    catch (error) {

        resultDiv.className =
            'err';

        resultDiv.innerHTML =
            '❌ เรียก API ไม่สำเร็จ: '
            + error.message;

    }

}

</script>
"""