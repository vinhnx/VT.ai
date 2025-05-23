// auth-bridge.js
// Authentication bridge for VT.ai - handles Supabase auth integration with Chainlit

(function() {
	'use strict';

	// Initialize authentication state
	let isAuthenticated = false;
	let userInfo = null;
	let supabaseClient = null;

	console.log('VT.ai Auth Bridge - Initializing...');

	// Wait for DOM and Chainlit to be ready
	document.addEventListener('DOMContentLoaded', function() {
		initializeAuthBridge();
	});

	// Delete all auth cookies
	function clearAuthCookies() {
		const cookies = document.cookie.split(';');

		for (let cookie of cookies) {
			const cookieName = cookie.split('=')[0].trim();
			// Delete all Supabase and session related cookies
			if (cookieName.includes('supabase') ||
				cookieName.includes('sb-') ||
				cookieName.includes('auth') ||
				cookieName.includes('session')) {
				document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`;
			}
		}
		console.log('Auth Bridge - Cleared all auth cookies');
	}

	// Complete logout process
	async function handleLogout() {
		console.log('Auth Bridge - Starting logout process');

		try {
			// First get Supabase client
			const { createClient } = window.supabaseClient;
			if (!supabaseClient) {
				supabaseClient = createClient(
					window.SUPABASE_URL,
					window.SUPABASE_ANON_KEY
				);
			}

			// Call Supabase signOut
			if (supabaseClient) {
				await supabaseClient.auth.signOut();
				console.log('Auth Bridge - Supabase signOut successful');
			}

			// Clear all auth cookies
			clearAuthCookies();

			// Update auth state
			isAuthenticated = false;
			userInfo = null;

			// Notify Chainlit about logout
			if (window.chainlit) {
				window.chainlit.emit('auth_status_changed', {
					authenticated: false,
					timestamp: Date.now()
				});
			}

			// Store current URL for redirect back after login
			const currentUrl = window.location.href;
			if (currentUrl && !currentUrl.includes('/auth/login')) {
				sessionStorage.setItem('vtai_redirect_after_login', currentUrl);
			}

			// Redirect to login with stored return URL
			const redirectUrl = sessionStorage.getItem('vtai_redirect_after_login');
			const loginUrl = redirectUrl ?
				`http://localhost:3000/auth/login?redirect_uri=${encodeURIComponent(redirectUrl)}` :
				'http://localhost:3000/auth/login';

			// Force reload to clear any remaining state
			window.location.href = loginUrl;

		} catch (error) {
			console.error('Auth Bridge - Error during logout:', error);
			// Fallback - force redirect to login
			window.location.href = 'http://localhost:3000/auth/login';
		}
	}

	function initializeAuthBridge() {
		console.log('Auth Bridge - Setting up authentication handlers...');

		// Handle Chainlit logout events
		if (window.chainlit) {
			window.chainlit.on('logout', handleLogout);
		}

		// Check if we're already authenticated
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

	// --- Force redirect on logout marker message ---
	function observeLogoutMarker() {
		function checkAndRedirect() {
			const logoutMsg = Array.from(document.querySelectorAll('.cl-message, .message, .cl-message-content, .cl-chat-message'))
				.find(el => el.textContent && el.textContent.includes('__FORCE_LOGOUT__'));
			if (logoutMsg) {
				clearAuthCookies();
				window.location.replace('http://localhost:3000/auth/login');
			}
		}
		const chatRoot = document.querySelector('#chainlit-app') || document.body;
		const observer = new MutationObserver(checkAndRedirect);
		observer.observe(chatRoot, { childList: true, subtree: true });
		// Also check immediately in case the message is already present
		setTimeout(checkAndRedirect, 500);
	}
	document.addEventListener('DOMContentLoaded', observeLogoutMarker);

	// Check for token in URL on load
	handleTokenFromUrl();

	// Expose some functions globally for debugging
	window.VTAuthBridge = {
		logout: handleLogout,
		checkStatus: checkAuthenticationStatus,
		redirectToLogin: redirectToLogin,
		isAuthenticated: () => isAuthenticated,
		getUserInfo: () => userInfo
	};

	console.log('VT.ai Auth Bridge - Initialization complete');
})();
