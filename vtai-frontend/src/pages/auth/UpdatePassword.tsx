import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import useRedirectIfAuthenticated from '../../components/useRedirectIfAuthenticated';

const supabase = createClient(
	import.meta.env.VITE_SUPABASE_URL,
	import.meta.env.VITE_SUPABASE_ANON_KEY
);

/**
 * Update password page for VT.ai.
 * Allows users to set a new password after reset.
 */
const UpdatePassword: React.FC = () => {
	useRedirectIfAuthenticated();
	const navigate = useNavigate();
	const [password, setPassword] = useState('');
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);

	const handleUpdate = async (e: React.FormEvent) => {
		e.preventDefault();
		setLoading(true);
		setError(null);
		const { error } = await supabase.auth.updateUser({ password });
		if (error) {
			setError(error.message);
		} else {
			window.location.href = '/';
		}
		setLoading(false);
	};

	return (
		<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 80 }}>
			<h1>Set New Password</h1>
			<form onSubmit={handleUpdate} style={{ display: 'flex', flexDirection: 'column', width: 320 }}>
				<input
					type="password"
					placeholder="New Password"
					value={password}
					onChange={e => setPassword(e.target.value)}
					required
					style={{ margin: 8, padding: 12 }}
				/>
				<button type="submit" style={{ margin: 8, padding: 12 }} disabled={loading}>
					{loading ? 'Processing...' : 'Update password'}
				</button>
			</form>
			{error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
		</div>
	);
};

export default UpdatePassword;
