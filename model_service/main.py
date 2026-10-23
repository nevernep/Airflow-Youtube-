"""
main.py — YouTube Views Prediction Studio
=========================================
Workshop 2: Model Serving API

UI แบบ YouTube Studio Dashboard
ใช้ FastAPI + Apache Airflow + Random Forest
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from youtube import (
    router as youtube_router,
    youtube_model_file_info,
    YOUTUBE_FORM_HTML,
)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="YouTube Views Prediction API",
    description=(
        "Machine Learning Model Serving "
        "with Apache Airflow + FastAPI"
    ),
    version="2.0.0",
)

app.include_router(youtube_router)


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "youtube-views-model-api",
        "youtube_model": youtube_model_file_info(),
    }


# =========================================================
# Home
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    model_info = youtube_model_file_info()

    model_status = (
        "Model Ready"
        if model_info["exists"]
        else "Model Not Found"
    )

    model_modified = (
        model_info["last_modified"]
        if model_info["last_modified"]
        else "Waiting for deployment"
    )

    page_start = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
YouTube ML Studio
</title>


<style>

/* ======================================================
   RESET
====================================================== */

* {{
    box-sizing: border-box;
}}

html {{
    scroll-behavior: smooth;
}}

body {{
    margin: 0;

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;

    background: #0f0f0f;

    color: #f1f1f1;
}}


a {{
    color: inherit;
}}


/* ======================================================
   TOP BAR
====================================================== */

.topbar {{

    position: fixed;

    top: 0;
    left: 0;
    right: 0;

    height: 64px;

    background:
        rgba(15, 15, 15, .96);

    backdrop-filter:
        blur(16px);

    border-bottom:
        1px solid #282828;

    display: flex;

    align-items: center;

    justify-content:
        space-between;

    padding:
        0 24px;

    z-index: 1000;
}}


.top-left {{

    display: flex;

    align-items: center;

    gap: 18px;
}}


.menu-icon {{

    width: 24px;

    font-size: 20px;

    color: #aaa;
}}


.brand {{

    display: flex;

    align-items: center;

    gap: 10px;

    font-size: 19px;

    font-weight: 700;

    letter-spacing:
        -.4px;
}}


.youtube-logo {{

    width: 38px;

    height: 26px;

    background: #ff0033;

    border-radius: 7px;

    display: flex;

    align-items: center;

    justify-content: center;

    box-shadow:
        0 5px 18px
        rgba(255, 0, 51, .25);

    color: white;

    font-size: 13px;
}}


.brand-light {{

    color: #aaa;

    font-weight: 500;
}}


.top-status {{

    background: #17251b;

    color: #55d976;

    border:
        1px solid #25472e;

    border-radius: 20px;

    padding:
        7px 13px;

    font-size: 12px;

    font-weight: 600;

    display: flex;

    align-items: center;

    gap: 7px;
}}


.online-dot {{

    width: 7px;

    height: 7px;

    background: #4ade80;

    border-radius: 50%;

    box-shadow:
        0 0 8px #4ade80;
}}


/* ======================================================
   SIDEBAR
====================================================== */

.sidebar {{

    position: fixed;

    top: 64px;

    left: 0;

    bottom: 0;

    width: 230px;

    padding:
        20px 14px;

    background: #0f0f0f;

    border-right:
        1px solid #272727;

    overflow-y: auto;
}}


.menu-label {{

    font-size: 11px;

    color: #717171;

    text-transform:
        uppercase;

    letter-spacing:
        1px;

    margin:
        22px 12px
        8px;
}}


.nav-item {{

    height: 44px;

    padding:
        0 14px;

    margin:
        3px 0;

    border-radius:
        9px;

    display: flex;

    align-items: center;

    gap: 13px;

    color: #aaa;

    text-decoration: none;

    font-size: 14px;

    transition:
        all .18s ease;
}}


.nav-item:hover {{

    background: #242424;

    color: white;
}}


.nav-item.active {{

    background: #272727;

    color: white;

    font-weight: 600;
}}


.nav-icon {{

    width: 22px;

    text-align: center;

    font-size: 17px;
}}


/* ======================================================
   MAIN
====================================================== */

.main {{

    margin-left: 230px;

    padding:
        96px 34px
        50px;

    min-height: 100vh;
}}


.page-header {{

    display: flex;

    align-items:
        flex-end;

    justify-content:
        space-between;

    margin-bottom:
        28px;
}}


.page-title {{

    margin: 0;

    font-size: 27px;

    letter-spacing:
        -.7px;
}}


.page-subtitle {{

    color: #8f8f8f;

    margin-top: 7px;

    font-size: 13px;
}}


.live-badge {{

    padding:
        8px 12px;

    border-radius:
        8px;

    background:
        #1d1d1d;

    border:
        1px solid #303030;

    color: #aaa;

    font-size: 12px;
}}


/* ======================================================
   ANALYTICS CARDS
====================================================== */

.analytics-grid {{

    display: grid;

    grid-template-columns:
        repeat(
            4,
            minmax(0,1fr)
        );

    gap: 16px;

    margin-bottom:
        22px;
}}


.analytics-card {{

    background: #181818;

    border:
        1px solid #2b2b2b;

    border-radius:
        14px;

    padding:
        20px;

    min-height: 118px;

    transition:
        transform .2s ease,
        border-color .2s ease;
}}


.analytics-card:hover {{

    transform:
        translateY(-2px);

    border-color:
        #444;
}}


.analytics-title {{

    color: #8f8f8f;

    font-size: 12px;

    margin-bottom:
        12px;
}}


.analytics-value {{

    font-size: 24px;

    font-weight: 700;

    letter-spacing:
        -.5px;

    color: #fff;
}}


.analytics-small {{

    margin-top:
        7px;

    color: #686868;

    font-size: 11px;
}}


.green {{

    color: #4ade80;
}}


.red {{

    color: #ff4e45;
}}


/* ======================================================
   CONTENT GRID
====================================================== */

.content-grid {{

    display: grid;

    grid-template-columns:
        minmax(0, 1.5fr)
        minmax(290px, .65fr);

    gap: 20px;
}}


/* ======================================================
   PANEL
====================================================== */

.panel {{

    background: #181818;

    border:
        1px solid #2b2b2b;

    border-radius:
        14px;

    overflow: hidden;
}}


.panel-header {{

    padding:
        20px 22px;

    border-bottom:
        1px solid #292929;
}}


.panel-title {{

    margin: 0;

    font-size: 16px;

    font-weight: 650;
}}


.panel-description {{

    color: #858585;

    font-size: 12px;

    margin:
        7px 0 0;

    line-height:
        1.5;
}}


.panel-body {{

    padding:
        22px;
}}


/* ======================================================
   FORM OVERRIDE
====================================================== */

/*
youtube.py มี HTML Form อยู่แล้ว
ตรงนี้ใช้ CSS ทำให้ฟอร์มเดิมกลายเป็น Studio UI
*/

.panel-body > hr {{
    display: none;
}}


.panel-body > h1 {{

    display: none;
}}


.panel-body > p {{

    display: none;
}}


.field {{

    margin-bottom:
        15px;
}}


.field label {{

    display: block;

    margin-bottom:
        7px;

    color: #b7b7b7;

    font-size: 12px;

    font-weight: 600;
}}


input,
select {{

    width: 100%;

    height: 44px;

    padding:
        0 13px;

    background: #121212;

    border:
        1px solid #343434;

    color: white;

    border-radius:
        8px;

    font-size: 13px;

    outline: none;

    transition:
        .2s ease;
}}


input:hover,
select:hover {{

    border-color: #4a4a4a;
}}


input:focus,
select:focus {{

    border-color:
        #3ea6ff;

    box-shadow:
        0 0 0 2px
        rgba(
            62,
            166,
            255,
            .12
        );
}}


input[type="checkbox"] {{

    width: auto;

    height: auto;

    accent-color:
        #3ea6ff;
}}


/* ======================================================
   BUTTON
====================================================== */

button {{

    width: 100%;

    min-height:
        46px;

    margin-top:
        12px;

    border: 0;

    border-radius:
        8px;

    background:
        #ff0033;

    color: white;

    font-weight: 700;

    font-size: 14px;

    cursor: pointer;

    transition:
        all .2s ease;

    box-shadow:
        0 7px 20px
        rgba(
            255,
            0,
            51,
            .18
        );
}}


button:hover {{

    background:
        #e9002e;

    transform:
        translateY(-1px);
}}


button:active {{

    transform:
        scale(.995);
}}


/* ======================================================
   RESULT
====================================================== */

#youtube_result {{

    margin-top:
        20px;

    display: none;

    padding:
        20px;

    border-radius:
        10px;
}}


#youtube_result.ok {{

    display: block;

    background:
        linear-gradient(
            135deg,
            #18251d,
            #142019
        );

    border:
        1px solid #2c5236;
}}


#youtube_result.err {{

    display: block;

    background:
        #291719;

    border:
        1px solid #5c292c;

    color:
        #ff7b7b;
}}


#youtube_result.ok > div:first-child {{

    font-size:
        28px !important;

    font-weight:
        750 !important;

    color:
        #5ee27e;

    letter-spacing:
        -.7px;

    margin-bottom:
        10px;
}}


.meta {{

    color: #8a9a8d;

    font-size: 11px;

    margin-top:
        5px;
}}


/* ======================================================
   MODEL PANEL
====================================================== */

.model-panel {{

    padding: 22px;
}}


.model-heading {{

    font-size:
        16px;

    font-weight:
        650;

    margin-bottom:
        20px;
}}


.model-status {{

    display: flex;

    align-items: center;

    gap: 10px;

    padding:
        13px;

    border-radius:
        9px;

    background: #122018;

    border:
        1px solid #254730;

    color:
        #5cde7c;

    font-size:
        13px;

    margin-bottom:
        20px;
}}


.status-circle {{

    width: 9px;

    height: 9px;

    border-radius:
        50%;

    background:
        #4ade80;

    box-shadow:
        0 0 9px
        #4ade80;
}}


.info-row {{

    padding:
        14px 0;

    border-bottom:
        1px solid #282828;
}}


.info-row:last-of-type {{

    border-bottom: 0;
}}


.info-label {{

    color:
        #6f6f6f;

    font-size:
        10px;

    text-transform:
        uppercase;

    letter-spacing:
        .8px;

    margin-bottom:
        5px;
}}


.info-value {{

    font-size:
        13px;

    color:
        #d1d1d1;

    word-break:
        break-word;

    line-height:
        1.5;
}}


/* ======================================================
   QUICK LINKS
====================================================== */

.quick-title {{

    margin:
        25px 0
        10px;

    font-size:
        11px;

    color:
        #6f6f6f;

    text-transform:
        uppercase;

    letter-spacing:
        .8px;
}}


.quick-link {{

    display: flex;

    align-items: center;

    justify-content:
        space-between;

    height:
        42px;

    padding:
        0 12px;

    margin:
        7px 0;

    border-radius:
        8px;

    background:
        #202020;

    border:
        1px solid #2e2e2e;

    color:
        #c7c7c7;

    text-decoration:
        none;

    font-size:
        12px;

    transition:
        .2s ease;
}}


.quick-link:hover {{

    background:
        #292929;

    color:
        #fff;

    border-color:
        #414141;
}}


/* ======================================================
   MINI GRAPH
====================================================== */

.chart-card {{

    margin-top:
        18px;

    padding:
        16px;

    border-radius:
        10px;

    background:
        #121212;

    border:
        1px solid #282828;
}}


.chart-title {{

    color:
        #868686;

    font-size:
        11px;

    margin-bottom:
        16px;
}}


.chart {{

    height:
        70px;

    display: flex;

    align-items:
        flex-end;

    gap:
        6px;
}}


.bar {{

    flex: 1;

    background:
        linear-gradient(
            to top,
            #ff0033,
            #ff4f69
        );

    border-radius:
        3px 3px
        1px 1px;

    opacity:
        .75;
}}


.bar:hover {{

    opacity: 1;
}}


/* ======================================================
   RESPONSIVE
====================================================== */

@media (
    max-width: 1050px
) {{

    .analytics-grid {{

        grid-template-columns:
            repeat(2,1fr);
    }}


    .content-grid {{

        grid-template-columns:
            1fr;
    }}

}}


@media (
    max-width: 760px
) {{

    .sidebar {{

        display: none;
    }}


    .main {{

        margin-left: 0;

        padding:
            90px 16px
            40px;
    }}


    .topbar {{

        padding:
            0 16px;
    }}

}}


@media (
    max-width: 520px
) {{

    .analytics-grid {{

        grid-template-columns:
            1fr;
    }}


    .page-header {{

        align-items:
            flex-start;

        flex-direction:
            column;

        gap:
            14px;
    }}

}}

</style>

</head>


<body>


<!-- ===================================================
     TOP BAR
=================================================== -->

<header class="topbar">

    <div class="top-left">

        <div class="menu-icon">
            ☰
        </div>

        <div class="brand">

            <div class="youtube-logo">
                ▶
            </div>

            YouTube

            <span class="brand-light">
                ML Studio
            </span>

        </div>

    </div>


    <div class="top-status">

        <span class="online-dot"></span>

        API Online

    </div>

</header>



<!-- ===================================================
     SIDEBAR
=================================================== -->

<aside class="sidebar">

    <a
        class="nav-item active"
        href="/"
    >

        <span class="nav-icon">
            ▦
        </span>

        Dashboard

    </a>


    <a
        class="nav-item"
        href="#prediction"
    >

        <span class="nav-icon">
            ▶
        </span>

        Prediction

    </a>


    <div class="menu-label">

        Model

    </div>


    <a
        class="nav-item"
        href="/youtube/model-info"
        target="_blank"
    >

        <span class="nav-icon">
            ◈
        </span>

        Model Info

    </a>


    <a
        class="nav-item"
        href="/health"
        target="_blank"
    >

        <span class="nav-icon">
            ♥
        </span>

        Health

    </a>


    <div class="menu-label">

        Developer

    </div>


    <a
        class="nav-item"
        href="/docs"
        target="_blank"
    >

        <span class="nav-icon">
            ⚙
        </span>

        Swagger API

    </a>


    <a
        class="nav-item"
        href="/redoc"
        target="_blank"
    >

        <span class="nav-icon">
            ◫
        </span>

        API Docs

    </a>

</aside>



<!-- ===================================================
     MAIN
=================================================== -->

<main class="main">


<div class="page-header">

    <div>

        <h1 class="page-title">

            Channel Analytics

        </h1>

        <div class="page-subtitle">

            YouTube Views Prediction
            • Machine Learning Dashboard

        </div>

    </div>


    <div class="live-badge">

        Workshop 2 • ML Pipeline

    </div>

</div>



<!-- ===================================================
     ANALYTICS
=================================================== -->

<section class="analytics-grid">


    <div class="analytics-card">

        <div class="analytics-title">

            ML Model

        </div>

        <div class="analytics-value">

            Random Forest

        </div>

        <div class="analytics-small green">

            ● Production model

        </div>

    </div>



    <div class="analytics-card">

        <div class="analytics-title">

            Prediction Target

        </div>

        <div class="analytics-value">

            Views

        </div>

        <div class="analytics-small">

            YouTube video views

        </div>

    </div>



    <div class="analytics-card">

        <div class="analytics-title">

            Model Status

        </div>

        <div class="analytics-value green">

            {model_status}

        </div>

        <div class="analytics-small">

            current_model.pkl

        </div>

    </div>



    <div class="analytics-card">

        <div class="analytics-title">

            Pipeline

        </div>

        <div class="analytics-value">

            Airflow

        </div>

        <div class="analytics-small">

            Automated ML workflow

        </div>

    </div>


</section>



<!-- ===================================================
     MAIN GRID
=================================================== -->

<section class="content-grid">


<!-- Prediction -->

<section
    class="panel"
    id="prediction"
>

    <div class="panel-header">

        <h2 class="panel-title">

            Predict YouTube Views

        </h2>

        <p class="panel-description">

            Enter video metadata and engagement
            statistics to estimate total views
            using the latest deployed ML model.

        </p>

    </div>


    <div class="panel-body">
"""

    page_end = f"""

    </div>

</section>



<!-- ===================================================
     MODEL INFORMATION
=================================================== -->

<aside class="panel">

    <div class="model-panel">

        <div class="model-heading">

            Model Performance

        </div>


        <div class="model-status">

            <span class="status-circle"></span>

            {model_status}

        </div>



        <div class="info-row">

            <div class="info-label">

                Algorithm

            </div>

            <div class="info-value">

                RandomForestRegressor

            </div>

        </div>



        <div class="info-row">

            <div class="info-label">

                Model Name

            </div>

            <div class="info-value">

                youtube_views_rf

            </div>

        </div>



        <div class="info-row">

            <div class="info-label">

                Features

            </div>

            <div class="info-value">

                10 Input Features

            </div>

        </div>



        <div class="info-row">

            <div class="info-label">

                Model Updated

            </div>

            <div class="info-value">

                {model_modified}

            </div>

        </div>



        <div class="info-row">

            <div class="info-label">

                Serving

            </div>

            <div class="info-value">

                FastAPI

            </div>

        </div>



        <!-- Mini Chart -->

        <div class="chart-card">

            <div class="chart-title">

                Prediction Activity

            </div>

            <div class="chart">

                <div
                    class="bar"
                    style="height:28%"
                ></div>

                <div
                    class="bar"
                    style="height:42%"
                ></div>

                <div
                    class="bar"
                    style="height:35%"
                ></div>

                <div
                    class="bar"
                    style="height:62%"
                ></div>

                <div
                    class="bar"
                    style="height:54%"
                ></div>

                <div
                    class="bar"
                    style="height:78%"
                ></div>

                <div
                    class="bar"
                    style="height:65%"
                ></div>

                <div
                    class="bar"
                    style="height:88%"
                ></div>

                <div
                    class="bar"
                    style="height:72%"
                ></div>

                <div
                    class="bar"
                    style="height:96%"
                ></div>

            </div>

        </div>



        <div class="quick-title">

            Developer Tools

        </div>


        <a
            class="quick-link"
            href="/health"
            target="_blank"
        >

            Health Check

            <span>
                ↗
            </span>

        </a>


        <a
            class="quick-link"
            href="/docs"
            target="_blank"
        >

            Swagger API

            <span>
                ↗
            </span>

        </a>


        <a
            class="quick-link"
            href="/youtube/model-info"
            target="_blank"
        >

            Model Information

            <span>
                ↗
            </span>

        </a>


    </div>

</aside>


</section>


</main>


</body>

</html>
"""

    return (
        page_start
        + YOUTUBE_FORM_HTML
        + page_end
    )