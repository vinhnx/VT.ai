import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
	import.meta.env.VITE_SUPABASE_URL,
	import.meta.env.VITE_SUPABASE_ANON_KEY
);

/**
 * Redirects authenticated users to the main app page.
 * Should be used at the top of auth-related pages.
 */
const useRedirectIfAuthenticated = () => {
	const navigate = useNavigate();
	useEffect(() => {
		const checkSession = async () => {
			const { data } = await supabase.auth.getSession();
			if (data.session && data.session.user) {
				navigate('/');
			}
		};
		checkSession();
	}, [navigate]);
};

export default useRedirectIfAuthenticated;
