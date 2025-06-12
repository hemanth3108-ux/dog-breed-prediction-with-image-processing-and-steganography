#!/bin/bash
echo "Installing requirements..."
pip install -r requirements.txt
echo "Running app..."
uvicorn main:app --reload
