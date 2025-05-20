#!/bin/bash
# Test connection between frontend and backend for VT.ai

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Testing VT.ai Frontend/Backend Connectivity${NC}\n"

# Check if the backend is running
echo "Testing backend health endpoint..."
backend_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health)

if [ "$backend_health" = "200" ]; then
    echo -e "${GREEN}✓ Backend is running (HTTP 200)${NC}"
else
    echo -e "${RED}✗ Backend is not responding correctly (HTTP $backend_health)${NC}"
    echo "  Make sure the backend server is running with: ./scripts/run_vtai_api.sh"
fi

# Check if the frontend can access the backend through its proxy
echo -e "\nTesting frontend proxy to backend..."
frontend_port=$(curl -s http://localhost:5173 2>/dev/null || curl -s http://localhost:5174 2>/dev/null || curl -s http://localhost:5175 2>/dev/null || echo "No frontend")

if [ "$frontend_port" = "No frontend" ]; then
    echo -e "${RED}✗ Frontend server not detected on common ports (5173, 5174, 5175)${NC}"
    echo "  Make sure the frontend server is running with: ./scripts/run_vtai_frontend.sh"
else
    # Try to detect the frontend port
    for port in 5173 5174 5175; do
        if curl -s "http://localhost:$port" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Frontend detected on port $port${NC}"

            # Test the proxy
            proxy_health=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/api/health")

            if [ "$proxy_health" = "200" ]; then
                echo -e "${GREEN}✓ Frontend proxy to backend is working (HTTP 200)${NC}"
            else
                echo -e "${RED}✗ Frontend proxy to backend failed (HTTP $proxy_health)${NC}"
                echo "  Check your Vite configuration in vite.config.ts"
            fi

            break
        fi
    done
fi

echo -e "\n${YELLOW}Troubleshooting Tips:${NC}"
echo "1. Make sure both servers are running"
echo "2. Check CORS settings in the backend"
echo "3. Verify proxy configuration in vite.config.ts"
echo "4. Check browser console for network errors"
echo "5. Try accessing the check.html page: http://localhost:5173/check.html"
