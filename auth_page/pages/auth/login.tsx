"""
Custom login page for VT.ai using Supabase social authentication.
On successful login, upserts user profile and redirects to VT.ai frontend.

Follows Google Python Style Guide for docstrings and project conventions.
"""
import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { supabase } from '../../lib/supabaseClient';
import { upsertUserProfile } from '../../lib/upsertUserProfile';

/**
 * Login page component for social authentication.
 * Redirects to VT.ai frontend on success.
 */
export default function Login() {
	const router = useRouter();

	useEffect(() => {
		const { data: { subscription } } = supabase.auth.onAuthStateChange(async (event, session) => {
			if (event === 'SIGNED_IN' && session?.user) {
				await upsertUserProfile(session.user);
				window.location.href = 'http://localhost:5173/';
			}
		});
		return () => subscription.unsubscribe();
	}, []);

	const handleSignIn = async (provider: string) => {
		await supabase.auth.signInWithOAuth({
			provider,
			options: {
				redirectTo: typeof window !== 'undefined' ? window.location.origin + '/auth/login' : undefined,
			},
		});
	};

	return (
		<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 80 }}>
			<h1>Sign in to VT.ai</h1>
			<button onClick={() => handleSignIn('github')} style={{ margin: 8, padding: 12 }}>Continue with GitHub</button>
			<button onClick={() => handleSignIn('google')} style={{ margin: 8, padding: 12 }}>Continue with Google</button>
		</div>
	);
}
