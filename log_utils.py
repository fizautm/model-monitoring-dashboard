import os
import pandas as pd
from datetime import datetime

LOG_PATH = "monitoring_logs.csv"

def log_prediction(model_version, model_type, input_summary,
                   prediction, latency_ms, feedback_score, feedback_text):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "model_type": model_type,
        "input_summary": input_summary,
        "prediction": prediction,
        "latency_ms": latency_ms,
        "feedback_score": feedback_score,
        "feedback_text": feedback_text
    }

    df_new = pd.DataFrame([row])

    if not os.path.exists(LOG_PATH):
        df_new.to_csv(LOG_PATH, index=False)
    else:
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)
