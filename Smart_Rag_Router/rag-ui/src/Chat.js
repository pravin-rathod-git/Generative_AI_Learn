import React, { useState } from "react";
import axios from "axios";

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", text: input };
    setMessages(prev => [...prev, userMessage]);

    try {
      const res = await axios.post("http://127.0.0.1:8000/chat", {
        query: input
      });

      const aiMessage = {
        role: "ai",
        text: res.data.answer,
        route: res.data.route
      };

      setMessages(prev => [...prev, aiMessage]);

    } catch (err) {
      console.error(err);
    }

    setInput("");
  };

  return (
    <div className="chat-container">
      <h2>🧠 Smart RAG Assistant</h2>

      <div className="chat-box">
        {messages.map((msg, idx) => (
          <div key={idx} className={msg.role}>
            <p>{msg.text}</p>
            {msg.route && <span className="route">({msg.route})</span>}
          </div>
        ))}
      </div>

      <div className="input-box">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default Chat;