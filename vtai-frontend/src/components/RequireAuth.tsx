import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
	import.meta.env.VITE_SUPABASE_URL,
	import.meta.env.VITE_SUPABASE_ANON_KEY
);

const RequireAuth: React.FC<{ children: React.ReactNode }> = ({ children }) => {
	const navigate = useNavigate();
	useEffect(() => {
		const checkSession = async () => {
			const { data } = await supabase.auth.getSession();
			if (!data.session || !data.session.user) {
				navigate('/auth/login');
			}
		};
		checkSession();
	}, [navigate]);
	return <>{children}</>;
};

export default RequireAuth;