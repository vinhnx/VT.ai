import type { AppProps } from 'next/app';
import dynamic from 'next/dynamic';
import '../styles/globals.css';

const ThemeSwitcher = dynamic(() => import('../components/ThemeSwitcher').then(mod => mod.ThemeSwitcher), { ssr: false });

function MyApp({ Component, pageProps }: AppProps) {
	return (
		<>
			<ThemeSwitcher />
			<Component {...pageProps} />
		</>
	);
}

export default MyApp;