import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
	import.meta.env.VITE_SUPABASE_URL,
	import.meta.env.VITE_SUPABASE_ANON_KEY
);

/**
 * Login page for password-based authentication.
 * On success, redirects to VT.ai frontend root.
 */
const Login: React.FC = () => {
	const navigate = useNavigate();
	const [email, setEmail] = useState('');
	const [password, setPassword] = useState('');
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);

	const handleLogin = async (e: React.FormEvent) => {
		e.preventDefault();
		setLoading(true);
		setError(null);
		const { data, error } = await supabase.auth.signInWithPassword({ email, password });
		if (error) {
			setError(error.message);
		} else if (data.user) {
			// TODO: Upsert profile in Supabase DB here if needed
			window.location.href = '/';
		}
		setLoading(false);
	};

	return (
		<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 80 }}>
			<h1>Sign in to VT.ai</h1>
			<form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', width: 320 }}>
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
					{loading ? 'Processing...' : 'Sign in'}
				</button>
			</form>
			{error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
		</div>
	);
};

export default Login;
