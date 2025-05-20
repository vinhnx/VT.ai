import React from 'react';
import { Link } from 'react-router-dom';

/**
 * AuthLogin page for VT.ai custom frontend.
 * Links to password-based auth routes.
 */
const AuthLogin: React.FC = () => {
	return (
		<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 80 }}>
			<h1>Sign in to VT.ai</h1>
			<Link to="/auth/login">
				<button style={{ margin: 8, padding: 12 }}>Sign in with Email</button>
			</Link>
			<Link to="/auth/sign-up">
				<button style={{ margin: 8, padding: 12 }}>Sign up</button>
			</Link>
			<Link to="/auth/forgot-password">
				<button style={{ margin: 8, padding: 12 }}>Forgot Password?</button>
			</Link>
		</div>
	);
};

export default AuthLogin;
