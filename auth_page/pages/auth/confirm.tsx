// Confirm email and password recovery page for Supabase password-based auth
import { supabase } from "@/lib/supabaseClient";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";

/**
 * ConfirmPage handles email confirmation and password recovery via Supabase Auth.
 * Redirects to update-password if recovery, or shows confirmation message.
 * @returns JSX.Element
 */
export default function ConfirmPage() {
	const router = useRouter();
	const [status, setStatus] = useState<string>("pending");
	const [error, setError] = useState<string>("");

	useEffect(() => {
		const { query } = router;
		const type = query.type as string | undefined;
		const access_token = query.access_token as string | undefined;
		const refresh_token = query.refresh_token as string | undefined;

		if (type === "recovery" && access_token && refresh_token) {
			// Set session and redirect to update-password
			supabase.auth.setSession({ access_token, refresh_token })
				.then(() => {
					setStatus("redirecting");
					router.replace("/auth/update-password");
				})
				.catch((err) => {
					setError("Failed to set session for password recovery.");
					setStatus("error");
				});
		} else if (type === "signup" || type === "email_confirmed") {
			setStatus("confirmed");
		} else {
			setStatus("unknown");
		}
	}, [router]);

	if (status === "redirecting") {
		return (
			<div className="flex flex-col items-center justify-center min-h-screen">
				<p>Redirecting to update password...</p>
			</div>
		);
	}
	if (status === "confirmed") {
		return (
			<div className="flex flex-col items-center justify-center min-h-screen">
				<h2 className="text-2xl font-bold mb-4">Email Confirmed</h2>
				<p className="mb-2">Your email has been confirmed. You can now log in.</p>
			</div>
		);
	}
	if (status === "error") {
		return (
			<div className="flex flex-col items-center justify-center min-h-screen">
				<h2 className="text-2xl font-bold mb-4">Error</h2>
				<p className="mb-2 text-red-600">{error}</p>
			</div>
		);
	}
	return (
		<div className="flex flex-col items-center justify-center min-h-screen">
			<h2 className="text-2xl font-bold mb-4">Confirm your email</h2>
			<p className="mb-2">Please check your email for a confirmation link.</p>
		</div>
	);
}
