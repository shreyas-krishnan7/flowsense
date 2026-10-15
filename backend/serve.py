from fastapi import FastAPI
from pydantic import BaseModel
from inference import FlowClassifier

app = FastAPI(title="FlowSense")
clf = FlowClassifier()

class FlowRequest(BaseModel):
    packets: list[list[float]]   # [[size_bytes, iat_ms, direction], ...] up to 30 packets

@app.post("/classify")
def classify(req: FlowRequest):
    return clf.classify([req.packets])[0]

@app.get("/health")
def health(): return {"status": "ok", "classes": len(clf.names)}
