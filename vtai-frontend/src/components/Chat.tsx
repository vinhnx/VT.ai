import React, { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const Chat: React.FC = () => {
	const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
	const [input, setInput] = useState('');
	const [loading, setLoading] = useState(false);

	const sendMessage = async (e: React.FormEvent) => {
		e.preventDefault();
		if (!input.trim()) return;
		const userMessage = { role: 'user', content: input };
		setMessages((prev) => [...prev, userMessage]);
		setInput('');
		setLoading(true);
		try {
			const res = await fetch(`${API_BASE_URL}/api/chat`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ message: input }),
			});
			const data = await res.json();
			setMessages((prev) => [...prev, { role: 'assistant', content: data.reply }]);
		} catch (err) {
			setMessages((prev) => [...prev, { role: 'assistant', content: 'Error: Unable to get response.' }]);
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="chat-container">
			<div className="messages">
				{messages.map((msg, idx) => (
					<div key={idx} className={`message ${msg.role}`}>
						<strong>{msg.role === 'user' ? 'You' : 'AI'}:</strong> {msg.content}
					</div>
				))}
				{loading && <div className="message assistant">AI is typing...</div>}
			</div>
			<form className="chat-input" onSubmit={sendMessage}>
				<input
					type="text"
					value={input}
					onChange={(e) => setInput(e.target.value)}
					placeholder="Type your message..."
					disabled={loading}
				/>
				<button type="submit" disabled={loading || !input.trim()}>Send</button>
			</form>
		</div>
	);
};

export default Chat;
