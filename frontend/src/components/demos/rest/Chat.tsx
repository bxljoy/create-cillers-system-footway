import React, { useState } from 'react';
import { ApiClientRest } from '../../../rest/api_client_rest'
import { createChatApi, Message } from '../../../rest/modules/chat'

interface ItemsProps {
    client: ApiClientRest
}

const Chat: React.FC<ItemsProps> = ({ client }) => {
  const chatApi = createChatApi(client);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    try {
      const botMessage = await chatApi.sendMessage(userMessage.content);
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: 'bot', content: `Error: Unable to fetch response: ${err}` },
      ]);
    }
  };

  return (
    <div className="flex-1 flex flex-col">
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-4 flex flex-col">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`p-3 rounded ${
                message.role === 'user'
                  ? 'bg-primary text-primary-content self-end'
                  : 'bg-base-200'
              }`}
              style={{ maxWidth: '70%', alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start' }}
            >
              {message.content}
            </div>
          ))}
        </div>
      </div>
      <div className="p-4 bg-base-300">
        <div className="form-control">
          <div className="input-group flex">
            <input
              type="text"
              className="input flex-grow"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button className="btn btn-primary" onClick={sendMessage}>
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
