// Logout page for Supabase password-based auth
import { supabase } from "@/lib/supabaseClient";
import { useRouter } from "next/router";
import { useEffect } from "react";

/**
 * LogoutPage signs the user out using Supabase Auth and redirects to login.
 * @returns JSX.Element
 */
export default function LogoutPage() {
	const router = useRouter();

	useEffect(() => {
		supabase.auth.signOut().then(() => {
			router.push("/auth/login");
		});
	}, [router]);

	return (
		<div className="flex flex-col items-center justify-center min-h-screen">
			<p>Signing out...</p>
		</div>
	);
}
