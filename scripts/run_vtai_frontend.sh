#!/bin/bash
# Run the VT.ai custom frontend (Vite dev server)
cd "$(dirname "$0")/../vtai-frontend"
npm install
npm run dev
