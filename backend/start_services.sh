#!/bin/bash

# Kill processes running on ports 8000 and 5173
fuser -k 8000/tcp
fuser -k 5173/tcp

# Start backend service
./start.sh &

# Navigate to frontend directory and start frontend service
cd ../frontend
npm run dev &
