#!/usr/bin/env python3
"""
Minimal test server to verify our changes work
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Test server working"}

@app.get("/")
async def root():
    return {"message": "Test server", "status": "running"}

if __name__ == "__main__":
    print("Starting test server...")
    uvicorn.run(app, host="127.0.0.1", port=8004, log_level="info")
