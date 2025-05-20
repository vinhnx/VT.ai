"""
Custom password-based auth page for VT.ai using Supabase.
On successful login or signup, upserts user profile and redirects to VT.ai frontend.

Follows Google Python Style Guide for docstrings and project conventions.
"""
import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { supabase } from '../../lib/supabaseClient';
import { upsertUserProfile } from '../../lib/upsertUserProfile';

/**
 * Password-based authentication page for VT.ai.
 * Handles login and signup, upserts user profile, and redirects on success.
 */
const PasswordAuth: React.FC = () => {
	const router = useRouter();
	const [email, setEmail] = useState('');
	const [password, setPassword] = useState('');
	const [isSignUp, setIsSignUp] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);

	const handleAuth = async (e: React.FormEvent) => {
		e.preventDefault();
		setLoading(true);
		setError(null);
		try {
			let result;
			if (isSignUp) {
				result = await supabase.auth.signUp({ email, password });
			} else {
				result = await supabase.auth.signInWithPassword({ email, password });
			}
			if (result.error) {
				setError(result.error.message);
			} else if (result.data?.user) {
				await upsertUserProfile(result.data.user);
				window.location.href = 'http://localhost:5173/';
			} else {
				setError('Check your email for confirmation or try again.');
			}
		} catch (err: any) {
			setError(err.message || 'Unknown error');
		} finally {
			setLoading(false);
		}
	};

	return (
		<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 80 }}>
			<h1>{isSignUp ? 'Sign up' : 'Sign in'} to VT.ai</h1>
			<form onSubmit={handleAuth} style={{ display: 'flex', flexDirection: 'column', width: 320 }}>
				<input
					type="email"
					placeholder="Email"
					value={email}
					onChange={e => setEmail(e.target.value)}
					required
					style={{ margin: 8, padding: 12 }}
				/>
				<input
					type="password"
					placeholder="Password"
					value={password}
					onChange={e => setPassword(e.target.value)}
					required
					style={{ margin: 8, padding: 12 }}
				/>
				<button type="submit" style={{ margin: 8, padding: 12 }} disabled={loading}>
					{loading ? 'Processing...' : isSignUp ? 'Sign up' : 'Sign in'}
				</button>
			</form>
			<button onClick={() => setIsSignUp(!isSignUp)} style={{ margin: 8, padding: 8, background: 'none', border: 'none', color: '#0070f3', cursor: 'pointer' }}>
				{isSignUp ? 'Already have an account? Sign in' : "Don't have an account? Sign up"}
			</button>
			{error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
		</div>
	);
};

export default PasswordAuth;
