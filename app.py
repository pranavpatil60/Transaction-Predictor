import io
import pandas as pd
import numpy as np
import pickle
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import ClientDisconnect  


# ---------------- APP INIT ----------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variable to store selected transactors for CSV download
global_export_df = pd.DataFrame()

# ---------------- LOAD MODEL FILES ----------------
try:
    with open("lgbm_final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
    with open("feature_order.pkl", "rb") as f:
        feature_order = pickle.load(f)
    print("✅ Model & Feature Configuration Loaded")
except Exception:
    print("❌ Model load failed")
    traceback.print_exc()
    model, threshold, feature_order = None, 0.5, []

# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    global global_export_df
    try:
        form = await request.form()
        file = form.get("file")

        if not file or not file.filename.endswith(".csv"):
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "❌ Please upload a valid CSV"}
            )

        df = pd.read_csv(file.file)
        if df.empty:
            raise ValueError("CSV is empty")

        # Handle ID column
        id_col = next((c for c in df.columns if "id" in c.lower()), None)
        df["USER_ID"] = df[id_col].astype(str) if id_col else [f"USR-{1000+i}" for i in range(len(df))]

        # ---------------- Feature Alignment (Optimized) ----------------
        missing_cols = set(feature_order) - set(df.columns)
        if missing_cols:
            df = pd.concat(
                [df, pd.DataFrame(0, index=df.index, columns=list(missing_cols))],
                axis=1
            )
        df = df.copy()  # de-fragment dataframe

        df_model = df[feature_order].apply(pd.to_numeric, errors="coerce").fillna(0)

        # ---------------- Prediction Logic ----------------
        try:
            proba = model.predict_proba(df_model)[:, 1]
        except:
            proba = model.predict(df_model)

        df["Confidence"] = (proba * 100).round(2)

        # Decision logic
        df["Decision"] = np.where(
            proba >= threshold, "1 (WILL TRANSACTION)",
            np.where(proba < 0.2, "0 (NO TRANSACTION)", "RE-VERIFY")
        )

        # Filter List for Download (Target 1 only)
        global_export_df = df[df["Decision"] == "1 (WILL TRANSACTION)"].copy()

        # Stats calculation
        total = len(df)
        will_transact = len(global_export_df)
        no_transact = (df["Decision"] == "0 (NO TRANSACTION)").sum()
        reverify = (df["Decision"] == "RE-VERIFY").sum()

        data_list = [
            {"id": r["USER_ID"], "conf": f"{r['Confidence']}%", "status": r["Decision"]}
            for _, r in df.iterrows()
        ]

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": True,
            "data": data_list,
            "total": total,
            "will_transact": will_transact,
            "no_transact": no_transact,
            "reverify": reverify,
            "threshold": threshold
        })

    except ClientDisconnect:
        print("⚠️ Client disconnected before upload completed")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "❌ Upload interrupted. Please try again."}
        )

    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"❌ {str(e)}"}
        )

@app.get("/download")
async def download_results():
    if global_export_df.empty:
        return {"error": "No transactors found to download"}

    stream = io.StringIO()
    global_export_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predicted_transactors.csv"
    return response
