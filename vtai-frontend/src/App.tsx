import React from 'react';
import Chat from './components/Chat';
import Header from './components/Header';

const App: React.FC = () => {
	return (
		<div className="app-container">
			<Header />
			<main className="main-content">
				<Chat />
			</main>
		</div>
	);
};

export default App;
