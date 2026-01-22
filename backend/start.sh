#!/usr/bin/env bash
set -o errexit

uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8001}"