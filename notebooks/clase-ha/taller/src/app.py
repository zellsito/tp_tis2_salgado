from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
import time
import asyncio
import socket

app = FastAPI()

@app.get("/sync/fibonacci/{n}")
def fibonacci(n: int):
    if n < 0:
        raise HTTPException(status_code=400, detail="Number must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

@app.get("/async/fibonacci/{n}")
async def async_fibonacci(n: int):
    return fibonacci(n)

@app.get("/async/sleep/{seconds}")
async def async_sleep(seconds: float):
    print(f"Sleeping for {seconds} seconds")
    if seconds < 0:
        raise HTTPException(status_code=400, detail="Sleep time must be non-negative")
    if seconds > 300:  # Limiting maximum sleep time to 5 minutes
        raise HTTPException(status_code=400, detail="Sleep time cannot exceed 300 seconds")
    await asyncio.sleep(seconds)
    return {"message": f"Slept for {seconds} seconds"}

@app.get("/sync/sleep/{seconds}")
def sleep(seconds: float):
    print(f"Sleeping for {seconds} seconds")
    if seconds < 0:
        raise HTTPException(status_code=400, detail="Sleep time must be non-negative")
    if seconds > 300:  # Limiting maximum sleep time to 5 minutes
        raise HTTPException(status_code=400, detail="Sleep time cannot exceed 300 seconds")
    time.sleep(seconds)
    return {"message": f"Slept for {seconds} seconds"}

@app.get("/whoami")
def whoami():
    hostname = socket.gethostname()
    return {"hostname": hostname}
