import React, { useState } from 'react';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
	import.meta.env.VITE_SUPABASE_URL,
	import.meta.env.VITE_SUPABASE_ANON_KEY
);

/**
 * Forgot password page for VT.ai.
 * Sends a password reset email using Supabase.
 */
const ForgotPassword: React.FC = () => {
	const [email, setEmail] = useState('');
	const [message, setMessage] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);

	const handleReset = async (e: React.FormEvent) => {
		e.preventDefault();
		setLoading(true);
		setError(null);
		setMessage(null);
		const { error } = await supabase.auth.resetPasswordForEmail(email);
		if (error) {
			setError(error.message);
		} else {
			setMessage('Check your email for a password reset link.');
		}
		setLoading(false);
	};

	return (
		<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 80 }}>
			<h1>Forgot Password</h1>
			<form onSubmit={handleReset} style={{ display: 'flex', flexDirection: 'column', width: 320 }}>
				<input
					type="email"
					placeholder="Email"
					value={email}
					onChange={e => setEmail(e.target.value)}
					required
					style={{ margin: 8, padding: 12 }}
				/>
				<button type="submit" style={{ margin: 8, padding: 12 }} disabled={loading}>
					{loading ? 'Processing...' : 'Send reset link'}
				</button>
			</form>
			{message && <div style={{ color: 'green', marginTop: 8 }}>{message}</div>}
			{error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
		</div>
	);
};

export default ForgotPassword;
