import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import useRedirectIfAuthenticated from './useRedirectIfAuthenticated';

const supabase = createClient(
	import.meta.env.VITE_SUPABASE_URL,
	import.meta.env.VITE_SUPABASE_ANON_KEY
);

/**
 * Combined authentication page for VT.ai.
 * Allows toggling between login and sign-up forms, with links to forgot/update password.
 */
const AuthLogin: React.FC = () => {
	useRedirectIfAuthenticated();

	const [isSignUp, setIsSignUp] = useState(false);
	const [email, setEmail] = useState('');
	const [password, setPassword] = useState('');
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const navigate = useNavigate();

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
				// Upsert user profile after successful auth
				await supabase.from('user_profiles').upsert({ id: result.data.user.id, email: result.data.user.email });
				window.location.href = '/';
			} else {
				setError('Check your email for confirmation or try again.');
			}
		} catch (err: any) {
			setError(err.message || 'Unknown error');
		} finally {
			setLoading(false);
		}
	};

	const handleSocial = async (provider: string) => {
		await supabase.auth.signInWithOAuth({ provider });
	};

	return (
		<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 80 }}>
			<h1>{isSignUp ? 'Sign up' : 'Sign in'} to VT.ai</h1>
			<div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
				<button onClick={() => setIsSignUp(false)} style={{ padding: 8, fontWeight: !isSignUp ? 'bold' : 'normal' }}>
					Sign In
				</button>
				<button onClick={() => setIsSignUp(true)} style={{ padding: 8, fontWeight: isSignUp ? 'bold' : 'normal' }}>
					Sign Up
				</button>
			</div>
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
			<div style={{ margin: 8 }}>
				<Link to="/auth/forgot-password">Forgot Password?</Link>
				{' | '}
				<Link to="/auth/update-password">Update Password</Link>
			</div>
			<div style={{ margin: 8 }}>
				<button onClick={() => handleSocial('github')} style={{ margin: 4, padding: 12 }}>Continue with GitHub</button>
				<button onClick={() => handleSocial('google')} style={{ margin: 4, padding: 12 }}>Continue with Google</button>
			</div>
			{error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
		</div>
	);
};

export default AuthLogin;
