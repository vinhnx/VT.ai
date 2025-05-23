// auth-redirect.js - Handle auth redirects for VT.ai
(function() {
    'use strict';

    // Store original URL before logout
    window.onbeforeunload = function() {
        if (window.location.pathname !== '/auth/login') {
            sessionStorage.setItem('vtai_redirect_after_login', window.location.href);
        }
    };

    // Handle redirect back after login
    window.onload = function() {
        if (window.location.pathname === '/auth/login') {
            const redirectUrl = sessionStorage.getItem('vtai_redirect_after_login');
            if (redirectUrl) {
                const url = new URL(window.location.href);
                url.searchParams.set('redirect_uri', redirectUrl);
                window.history.replaceState({}, '', url.toString());
                sessionStorage.removeItem('vtai_redirect_after_login');
            }
        }
    };
})();