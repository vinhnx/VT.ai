// auth-bridge.js
// Authentication bridge for VT.ai - handles Supabase auth integration with Chainlit

(function() {
	'use strict';

	// Initialize authentication state
	let isAuthenticated = false;
	let userInfo = null;

	// Supabase configuration - these will be set via environment or server
	let supabaseUrl = null;
	let supabaseAnonKey = null;
	let supabaseClient = null;

	console.log('VT.ai Auth Bridge - Initializing...');

	// Wait for DOM and Chainlit to be ready
	document.addEventListener('DOMContentLoaded', function() {
		initializeAuthBridge();
	});

	function initializeAuthBridge() {
		console.log('Auth Bridge - Setting up authentication handlers...');

		// Check if we're already authenticated via server-side middleware
		checkAuthenticationStatus();

		// Set up periodic auth check
		setInterval(checkAuthenticationStatus, 30000); // Check every 30 seconds
	}

	function checkAuthenticationStatus() {
		// Check for auth token in cookies
		const authToken = getCookie('supabase-auth-token');

		if (authToken && !isAuthenticated) {
			console.log('Auth Bridge - Token found, validating...');
			validateToken(authToken);
		} else if (!authToken && isAuthenticated) {
			console.log('Auth Bridge - Token missing, user logged out');
			handleLogout();
		}
	}

	function getCookie(name) {
		const value = `; ${document.cookie}`;
		const parts = value.split(`; ${name}=`);
		if (parts.length === 2) return parts.pop().split(';').shift();
		return null;
	}

	function validateToken(token) {
		// For now, we'll trust the server-side validation
		// In the future, we could add client-side validation here
		isAuthenticated = true;
		console.log('Auth Bridge - Authentication validated');

		// Notify Chainlit about authentication status
		if (window.chainlit) {
			window.chainlit.emit('auth_status_changed', {
				authenticated: true,
				timestamp: Date.now()
			});
		}
	}

	function handleLogout() {
		isAuthenticated = false;
		userInfo = null;

		console.log('Auth Bridge - User logged out');

		// Notify Chainlit about logout
		if (window.chainlit) {
			window.chainlit.emit('auth_status_changed', {
				authenticated: false,
				timestamp: Date.now()
			});
		}

		// Redirect to login if needed
		// This is handled by server-side middleware, but we can add client-side logic here
	}

	// Function to handle login redirect with proper redirect_uri
	function redirectToLogin() {
		const currentUrl = encodeURIComponent(window.location.href);
		const loginUrl = `http://localhost:3000/auth/login?redirect_uri=${currentUrl}`;
		console.log('Auth Bridge - Redirecting to login:', loginUrl);
		window.location.href = loginUrl;
	}

	// Function to handle token from URL parameters (if login redirects with token)
	function handleTokenFromUrl() {
		const urlParams = new URLSearchParams(window.location.search);
		const token = urlParams.get('token') || urlParams.get('access_token');

		if (token) {
			console.log('Auth Bridge - Token found in URL, setting cookie...');

			// Set the token as a cookie
			document.cookie = `supabase-auth-token=${token}; path=/; SameSite=Lax`;

			// Clean up URL
			const cleanUrl = window.location.origin + window.location.pathname;
			window.history.replaceState({}, document.title, cleanUrl);

			// Validate the new token
			validateToken(token);
		}
	}

	// Check for token in URL on load
	handleTokenFromUrl();

	// Expose some functions globally for debugging
	window.VTAuthBridge = {
		checkStatus: checkAuthenticationStatus,
		redirectToLogin: redirectToLogin,
		isAuthenticated: () => isAuthenticated,
		getUserInfo: () => userInfo
	};

	console.log('VT.ai Auth Bridge - Initialization complete');
})();
