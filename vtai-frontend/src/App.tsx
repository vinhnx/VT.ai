import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AuthLogin from './components/AuthLogin';
import Chat from './components/Chat';
import Header from './components/Header';
import Login from './pages/auth/Login';
import SignUp from './pages/auth/SignUp';
import ForgotPassword from './pages/auth/ForgotPassword';
import UpdatePassword from './pages/auth/UpdatePassword';

const App: React.FC = () => {
	return (
		<Router>
			<div className="app-container">
				<Header />
				<main className="main-content">
					<Routes>
						<Route path="/auth/login" element={<Login />} />
						<Route path="/auth/sign-up" element={<SignUp />} />
						<Route path="/auth/forgot-password" element={<ForgotPassword />} />
						<Route path="/auth/update-password" element={<UpdatePassword />} />
						<Route path="/" element={<Chat />} />
					</Routes>
				</main>
			</div>
		</Router>
	);
};

export default App;
