# VT.ai Custom Frontend

This is a custom React-based frontend for the VT.ai application.

## Getting Started

1. Install dependencies:

    cd vtai-frontend
    npm install

2. Start the development server:

    npm run dev

3. Open [http://localhost:5173](http://localhost:5173) in your browser.

## Features

- Modern chat interface for AI interaction
- Custom branding and styles
- Easy integration with backend API

## Build for Production

    npm run build

## Connecting to the Backend

The frontend expects a backend API at `/api/chat`. By default, it connects to `http://localhost:8000`.

To change the backend URL, edit `.env.local`:

    VITE_API_BASE_URL=http://localhost:8000

## Running Everything Together

1. Start the backend API:

    cd ../vtai-server
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

2. In another terminal, start the frontend:

    cd vtai-frontend
    npm install
    npm run dev

3. Open [http://localhost:5173](http://localhost:5173) in your browser.

You can now chat with the AI using the custom frontend.

## Notes

- Update the `/api/chat` endpoint in `Chat.tsx` to match your backend.
- Place your logo in `public/logo_dark.png` if you want to customize branding.
