from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import pandas as pd

from analyzer.parser import parse_file
from analyzer.importance import score_columns
from analyzer.stats import compute_stats
from analyzer.charts import generate_chart_data

app = FastAPI(title="DataSage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = {"csv", "xlsx", "xls", "json"}
    ext = file.filename.split(".")[-1].lower()

    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")

    contents = await file.read()

    try:
        df = parse_file(contents, ext)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {str(e)}")

    # compute stats first
    stats = compute_stats(df)

    # importance now depends on stats
    column_importance = score_columns(stats)

    # charts depend on stats
    charts = generate_chart_data(df, stats)

    preview_df = df.head(10)
    preview = preview_df.where(pd.notnull(preview_df), None)

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "column_importance": column_importance,
        "stats": stats,
        "charts": charts,
        "preview": preview.to_dict(orient="records"),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)