import { createClient } from '@supabase/supabase-js';
import type { NextApiRequest, NextApiResponse } from 'next';

/**
 * API route for email sign-in using Supabase.
 * Accepts POST requests with { email, password } in the body.
 * Returns Supabase auth response.
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
	if (req.method !== 'POST') {
		return res.status(405).json({ error: 'Method not allowed' });
	}

	const { email, password } = req.body;
	if (!email || !password) {
		return res.status(400).json({ error: 'Email and password are required' });
	}

	const supabase = createClient(
		process.env.NEXT_PUBLIC_SUPABASE_URL!,
		process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
	);

	const { data, error } = await supabase.auth.signInWithPassword({ email, password });
	if (error) {
		return res.status(401).json({ error: error.message });
	}
	return res.status(200).json({ user: data.user, session: data.session });
}
