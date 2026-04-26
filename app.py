"""
COMPAS Bias Audit — FastAPI Web Server
=======================================
This file turns the bias audit project into a web API
that runs on Google Cloud Run.

Endpoints:
  GET /           — welcome + list of endpoints
  GET /health     — health check (always returns OK)
  GET /run-audit  — runs all 6 phases of the audit pipeline
  GET /results    — returns all fairness metrics as JSON
  GET /report     — downloads the audit report markdown file
  GET /explain    — Gemini AI generates a plain-English bias explanation
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess, os, json, pandas as pd

app = FastAPI(
    title="COMPAS Bias Audit API",
    description="End-to-end algorithmic bias detection and mitigation system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html><body style="font-family:sans-serif;max-width:600px;margin:40px auto;padding:20px">
    <h1 style="color:#1a73e8">COMPAS Bias Audit API</h1>
    <p style="color:#555">End-to-end algorithmic bias detection system — [Unbiased AI Decision] Challenge</p>
    <hr>
    <h3>Available Endpoints</h3>
    <ul>
      <li><a href="/health">/health</a> — Health check</li>
      <li><a href="/run-audit">/run-audit</a> — Run full 6-phase bias audit pipeline</li>
      <li><a href="/results">/results</a> — Get all fairness metrics as JSON</li>
      <li><a href="/report">/report</a> — Download full audit report</li>
      <li><a href="/explain">/explain</a> — Gemini AI bias explanation</li>
      <li><a href="/docs">/docs</a> — Interactive API documentation</li>
    </ul>
    <p style="color:#888;font-size:12px">Built by BiasGuard AI Team — P Stephen</p>
    </body></html>
    """


@app.get("/health")
def health():
    return {"status": "healthy", "project": "COMPAS Bias Audit", "version": "1.0.0"}


@app.get("/run-audit")
def run_audit():
    """Run all 6 phases of the COMPAS bias audit pipeline."""
    results = {}
    scripts = [
        "01_data_loading",
        "02_eda",
        "03_model_training",
        "04_bias_detection",
        "05_mitigation",
        "06_report_generator",
    ]

    for script in scripts:
        try:
            result = subprocess.run(
                ["python", f"src/{script}.py"],
                capture_output=True,
                text=True,
                timeout=180,
                encoding="utf-8",
                errors="replace"
            )
            if result.returncode != 0:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Phase {script} failed",
                        "detail": result.stderr[-500:] if result.stderr else "Unknown error",
                        "completed_phases": list(results.keys()),
                    }
                )
            results[script] = "completed"
        except subprocess.TimeoutExpired:
            return JSONResponse(
                status_code=500,
                content={"error": f"{script} timed out after 180 seconds"}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )

    return {
        "status": "success",
        "message": "All 6 phases completed successfully!",
        "phases": results,
        "next_steps": [
            "Visit /results for fairness metrics",
            "Visit /explain for Gemini AI explanation",
            "Visit /report to download audit report"
        ]
    }


@app.get("/results")
def get_results():
    """Return all bias audit metrics as JSON."""
    try:
        fairness  = pd.read_csv("output/metrics/fairness_metrics.csv")
        per_group = pd.read_csv("output/metrics/per_group_metrics.csv")
        mitigation = pd.read_csv("output/metrics/mitigation_results.csv")
        model_perf = pd.read_csv("output/metrics/model_performance.csv")

        return {
            "status": "success",
            "summary": {
                "total_defendants": 6172,
                "black_fpr": float(per_group[per_group["group"] == "African-American"]["FPR"].values[0]),
                "white_fpr": float(per_group[per_group["group"] == "Caucasian"]["FPR"].values[0]),
                "disparate_impact": float(fairness["disparate_impact"].values[0]),
                "bias_detected": float(fairness["disparate_impact"].values[0]) < 0.80,
            },
            "fairness_metrics": fairness.to_dict(orient="records")[0],
            "per_group_metrics": per_group.to_dict(orient="records"),
            "mitigation_results": mitigation.to_dict(orient="records"),
            "model_performance": model_perf.to_dict(orient="records"),
        }
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": "Results not found. Please run /run-audit first."}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/report")
def get_report():
    """Download the generated audit report."""
    report_path = "report/audit_report.md"
    if os.path.exists(report_path):
        return FileResponse(
            report_path,
            media_type="text/markdown",
            filename="COMPAS_Bias_Audit_Report.md"
        )
    return JSONResponse(
        status_code=404,
        content={"error": "Report not found. Please run /run-audit first."}
    )


@app.get("/explain")
def explain_bias():
    """Use Google Gemini AI to explain bias findings in plain English."""
    gemini_key = os.environ.get("AIzaSyAEc4kha2TdjMqD7JsnaXuAdtU3jMUsGuM")
    if not gemini_key:
        return JSONResponse(
            status_code=400,
            content={"error": "GEMINI_API_KEY environment variable not set."}
        )

    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Load metrics
        fairness  = pd.read_csv("output/metrics/fairness_metrics.csv").to_dict(orient="records")[0]
        per_group = pd.read_csv("output/metrics/per_group_metrics.csv")
        aa_row = per_group[per_group["group"] == "African-American"].iloc[0]
        ca_row = per_group[per_group["group"] == "Caucasian"].iloc[0]

        prompt = f"""
You are an AI ethics expert reviewing algorithmic bias in the US criminal justice system.

The COMPAS recidivism algorithm (used by courts to decide bail and sentencing) was audited
with these results:

BIAS METRICS:
- Black defendants False Positive Rate: {aa_row['FPR']*100:.1f}%
- White defendants False Positive Rate: {ca_row['FPR']*100:.1f}%
- Disparate Impact ratio: {fairness['disparate_impact']:.3f} (1.0 = fair, below 0.80 = biased)
- Equal Opportunity Difference: {fairness['equal_opp_diff']:.3f}
- Statistical Parity Difference: {fairness['stat_parity_diff']:.3f}

Write a 4-sentence plain-English explanation of:
1. What the bias finding means
2. Who is being harmed and how
3. Why this bias exists in the data
4. What should be done to fix it

Write clearly for a non-technical audience. Start directly with the finding.
"""

        response = model.generate_content(prompt)
        return {
            "status": "success",
            "gemini_explanation": response.text,
            "metrics_used": {
                "black_fpr": f"{aa_row['FPR']*100:.1f}%",
                "white_fpr": f"{ca_row['FPR']*100:.1f}%",
                "disparate_impact": fairness['disparate_impact'],
            }
        }

    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": "Metrics not found. Please run /run-audit first."}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
