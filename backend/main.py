from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from model_service import load_and_predict_timeline, run_ablation_study
import uvicorn

app = FastAPI(title="Financial Anomaly Detection API")

# Setup CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/timeline")
def get_timeline(
    dataset: str = Query("5y"),
    q_threshold: float = Query(0.98),
    vol_q_low: float = Query(0.33),
    vol_q_high: float = Query(0.66),
    use_journal: bool = Query(False)
):
    """Returns the full parsed timeline, anomalies, and metrics."""
    data = load_and_predict_timeline(
        dataset=dataset, 
        q_threshold=q_threshold, 
        vol_q_low=vol_q_low, 
        vol_q_high=vol_q_high,
        use_journal=use_journal
    )
    if isinstance(data, dict) and "error" in data:
        return {"status": "error", "message": data["error"]}
    return {"status": "success", "data": data}

@app.get("/api/ablation")
def get_ablation(
    dataset: str = Query("5y"),
    q_threshold: float = Query(0.98),
    vol_q_low: float = Query(0.33),
    vol_q_high: float = Query(0.66),
    use_journal: bool = Query(False)
):
    """Returns ablation study: MoE vs Single Model, per-expert metrics, and feature importances."""
    data = run_ablation_study(
        dataset=dataset,
        q_threshold=q_threshold,
        vol_q_low=vol_q_low,
        vol_q_high=vol_q_high,
        use_journal=use_journal
    )
    if isinstance(data, dict) and "error" in data:
        return {"status": "error", "message": data["error"]}
    return {"status": "success", "data": data}

@app.get("/api/ping")
def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    print("🚀 Starting FastAPI Server on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
