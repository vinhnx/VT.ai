import type { AppProps } from 'next/app';
import { useEffect, useState } from 'react';
import '../styles/globals.css';

function ThemeSwitcher() {
	const [theme, setTheme] = useState<'light' | 'dark'>(
		(typeof window !== 'undefined' && localStorage.getItem('theme') === 'dark') ? 'dark' : 'light'
	);

	useEffect(() => {
		if (theme === 'dark') {
			document.documentElement.classList.add('dark');
			localStorage.setItem('theme', 'dark');
		} else {
			document.documentElement.classList.remove('dark');
			localStorage.setItem('theme', 'light');
		}
	}, [theme]);

	return (
		<button
			className="fixed top-4 right-4 z-50 rounded px-3 py-2 bg-card text-foreground border shadow"
			onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
		>
			{theme === 'dark' ? '🌙 Dark' : '☀️ Light'}
		</button>
	);
}

function MyApp({ Component, pageProps }: AppProps) {
	return (
		<>
			<ThemeSwitcher />
			<Component {...pageProps} />
		</>
	);
}

export default MyApp;
