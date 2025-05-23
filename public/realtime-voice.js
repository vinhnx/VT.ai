// realtime-voice.js
// Client-side handler for real-time voice conversation using WebRTC and OpenAI's Realtime API

(function() {
  // Global variables for WebRTC and WebSocket connections
  let mediaRecorder = null;
  let audioContext = null;
  let websocket = null;
  let recording = false;
  let sessionId = null;
  let apiKey = null;
  let websocketUrl = null;
  let currentModel = null;
  let currentVoice = null;
  let language = 'en';

  // Listen for custom events from the server
  document.addEventListener('DOMContentLoaded', function() {
    // Check if window.chainlit is available (it's loaded asynchronously)
    const checkChainlit = setInterval(() => {
      if (window.chainlit) {
        clearInterval(checkChainlit);
        setupEventListeners();
      }
    }, 100);
  });

  function setupEventListeners() {
    // Listen for custom events from the server
    window.chainlit.on('custom', async (data) => {
      // Handle different event types
      if (data.type === 'realtime_init') {
        await initializeRealtimeVoice(data);
      } else if (data.type === 'realtime_terminate') {
        terminateRealtimeVoice(data);
      } else if (data.type === 'realtime_toggle_recording') {
        toggleRecording(data);
      } else if (data.type === 'realtime_set_language') {
        setLanguage(data);
      } else if (data.type === 'realtime_set_voice') {
        setVoice(data);
      }
    });
  }

  async function initializeRealtimeVoice(data) {
    // Store session variables
    sessionId = data.sessionId;
    apiKey = data.apiKey;
    websocketUrl = data.websocketUrl;
// ...existing code...
