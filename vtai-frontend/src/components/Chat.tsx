import React, { useEffect, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import './Chat.css';

// Using relative URL when running in development to leverage Vite's proxy
const API_BASE_URL = '';

interface Message {
	id: string;
	role: 'user' | 'assistant' | 'system';
	content: string;
	timestamp: Date;
}

interface ChatError {
	type: 'api' | 'network' | 'quota' | 'auth' | 'unknown';
	message: string;
}

const Chat: React.FC = () => {
	const [sessionId, setSessionId] = useState<string>('');
	const [messages, setMessages] = useState<Message[]>([]);
	const [input, setInput] = useState('');
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<ChatError | null>(null);
	const messagesEndRef = useRef<HTMLDivElement>(null);

	// Initialize session ID on component mount
	useEffect(() => {
		const storedSessionId = localStorage.getItem('vtai_session_id');
		if (storedSessionId) {
			setSessionId(storedSessionId);
			// Load existing messages for this session
			const storedMessages = localStorage.getItem(`vtai_messages_${storedSessionId}`);
			if (storedMessages) {
				try {
					setMessages(JSON.parse(storedMessages));
				} catch (e) {
					console.error('Failed to parse stored messages:', e);
				}
			}
		} else {
			const newSessionId = uuidv4();
			setSessionId(newSessionId);
			localStorage.setItem('vtai_session_id', newSessionId);
		}
	}, []);

	// Save messages to localStorage whenever they change
	useEffect(() => {
		if (sessionId && messages.length > 0) {
			localStorage.setItem(`vtai_messages_${sessionId}`, JSON.stringify(messages));
		}
	}, [messages, sessionId]);

	// Auto-scroll to bottom when messages change
	useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	}, [messages]);

	const sendMessage = async (e: React.FormEvent) => {
		e.preventDefault();
		if (!input.trim()) return;

		// Clear previous errors
		setError(null);

		const messageId = uuidv4();
		const userMessage: Message = {
			id: messageId,
			role: 'user',
			content: input,
			timestamp: new Date()
		};

		setMessages((prev) => [...prev, userMessage]);
		setInput('');
		setLoading(true);

		try {
			const res = await fetch(`${API_BASE_URL}/api/chat`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'X-Session-ID': sessionId
				},
				body: JSON.stringify({
					message: input,
					session_id: sessionId,
					message_id: messageId
				}),
			});

			if (!res.ok) {
				let errorData;
				try {
					errorData = await res.json();
				} catch (e) {
					errorData = { error: 'Failed to parse error response' };
				}

				let errorType: ChatError['type'] = 'unknown';
				if (res.status === 401) errorType = 'auth';
				else if (res.status === 429) errorType = 'quota';

				throw new Error(errorData.error || `Error: ${res.status} (${errorType})`);
			}

			const data = await res.json();

			setMessages((prev) => [
				...prev,
				{
					id: data.id || uuidv4(),
					role: 'assistant',
					content: data.reply,
					timestamp: new Date()
				}
			]);
		} catch (err) {
			console.error('Chat API error:', err);

			let errorMessage = 'Error: Unable to get response.';
			let errorType: ChatError['type'] = 'network';

			if (err instanceof Error) {
				errorMessage = err.message;
				if (errorMessage.includes('quota')) errorType = 'quota';
				if (errorMessage.includes('auth')) errorType = 'auth';
			}

			setError({ type: errorType, message: errorMessage });

			setMessages((prev) => [
				...prev,
				{
					id: uuidv4(),
					role: 'assistant',
					content: errorMessage,
					timestamp: new Date()
				}
			]);
		} finally {
			setLoading(false);
		}
	};

	const startNewConversation = () => {
		const newSessionId = uuidv4();
		setSessionId(newSessionId);
		localStorage.setItem('vtai_session_id', newSessionId);
		setMessages([]);
		setError(null);
	};

	return (
		<div className="chat-container">
			<div className="chat-header">
				<h2>VT.ai Chat</h2>
				<button
					onClick={startNewConversation}
					className="new-chat-button"
					title="Start a new conversation"
				>
					New Chat
				</button>
			</div>

			{error && (
				<div className={`error-message error-${error.type}`}>
					{error.message}
					{error.type === 'quota' && (
						<p>The API key has reached its quota limit. Please try again later.</p>
					)}
				</div>
			)}

			<div className="messages-container">
				{messages.length === 0 ? (
					<div className="empty-chat">
						<p>Start a conversation with VT.ai</p>
					</div>
				) : (
					<div className="messages">
						{messages.map((msg) => (
							<div key={msg.id} className={`message ${msg.role}`}>
								<div className="message-header">
									<strong>{msg.role === 'user' ? 'You' : 'VT.ai'}</strong>
									<span className="message-time">
										{new Date(msg.timestamp).toLocaleTimeString()}
									</span>
								</div>
								<div className="message-content">{msg.content}</div>
							</div>
						))}
						{loading && (
							<div className="message assistant loading">
								<div className="message-header">
									<strong>VT.ai</strong>
								</div>
								<div className="message-content">
									<div className="typing-indicator">
										<span></span>
										<span></span>
										<span></span>
									</div>
								</div>
							</div>
						)}
						<div ref={messagesEndRef} />
					</div>
				)}
			</div>

			<form className="chat-input" onSubmit={sendMessage}>
				<input
					type="text"
					value={input}
					onChange={(e) => setInput(e.target.value)}
					placeholder="Type your message..."
					disabled={loading}
				/>
				<button
					type="submit"
					disabled={loading || !input.trim()}
					className="send-button"
				>
					{loading ? 'Sending...' : 'Send'}
				</button>
			</form>
		</div>
	);
};

export default Chat;
