#!/usr/bin/env bash
set -o errexit

# Render sets PORT automatically
uvicorn app.main:app --host 0.0.0.0 --port $PORT
