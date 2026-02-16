"""
Web Interface for YouAI
Serve your trained model with a beautiful web UI
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from inference import YouAIInference
import argparse

app = Flask(__name__)
CORS(app)

# Global model instance
youai_model = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouAI - Your Personal AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }

        .header h1 {
            font-size: 32px;
            margin-bottom: 8px;
        }

        .header p {
            font-size: 14px;
            opacity: 0.9;
        }

        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            margin-top: 8px;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }

        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }

        .message.user {
            justify-content: flex-end;
        }

        .message-bubble {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.5;
        }

        .message.user .message-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message.ai .message-bubble {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .message-label {
            font-size: 11px;
            color: #999;
            margin-bottom: 4px;
            padding: 0 5px;
        }

        .controls {
            padding: 15px 20px;
            background: #f0f0f0;
            border-top: 1px solid #ddd;
        }

        .control-row {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            align-items: center;
            font-size: 13px;
        }

        .control-row label {
            min-width: 120px;
            color: #666;
        }

        .control-row input[type="range"] {
            flex: 1;
        }

        .control-row .value {
            min-width: 40px;
            text-align: right;
            font-weight: bold;
            color: #667eea;
        }

        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
        }

        #userInput {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }

        #userInput:focus {
            border-color: #667eea;
        }

        #sendBtn {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: transform 0.2s;
        }

        #sendBtn:hover {
            transform: scale(1.05);
        }

        #sendBtn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .typing-indicator {
            display: none;
            padding: 10px 16px;
            background: white;
            border-radius: 18px;
            width: fit-content;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .typing-indicator span {
            height: 10px;
            width: 10px;
            background: #667eea;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-10px);
            }
        }

        .chat-container::-webkit-scrollbar {
            width: 8px;
        }

        .chat-container::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        .chat-container::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 4px;
        }

        .examples {
            padding: 10px 20px;
            background: #f9f9f9;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 8px;
            overflow-x: auto;
        }

        .example-btn {
            padding: 6px 12px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 15px;
            font-size: 12px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s;
        }

        .example-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 YouAI</h1>
            <p>Your personally trained AI assistant</p>
            <div class="badge">Trained from scratch • 2B parameters</div>
        </div>
        
        <div class="examples">
            <button class="example-btn" onclick="setPrompt('Tell me about yourself')">About you</button>
            <button class="example-btn" onclick="setPrompt('Explain quantum computing')">Quantum computing</button>
            <button class="example-btn" onclick="setPrompt('Write a short story')">Write a story</button>
            <button class="example-btn" onclick="setPrompt('What is machine learning?')">Machine learning</button>
        </div>
        
        <div class="controls">
            <div class="control-row">
                <label>Temperature:</label>
                <input type="range" id="temperature" min="0" max="2" step="0.1" value="0.8">
                <span class="value" id="tempValue">0.8</span>
            </div>
            <div class="control-row">
                <label>Max Length:</label>
                <input type="range" id="maxLength" min="20" max="200" step="10" value="100">
                <span class="value" id="lengthValue">100</span>
            </div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message ai">
                <div>
                    <div class="message-label">YouAI</div>
                    <div class="message-bubble">
                        Hey! I'm YouAI - your personally trained AI model. I was trained from scratch just for you! Ask me anything, and let's see what I've learned. 🚀
                    </div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        
        <div class="input-container">
            <input 
                type="text" 
                id="userInput" 
                placeholder="Type your message here..." 
                autocomplete="off"
            />
            <button id="sendBtn">Send</button>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const typingIndicator = document.getElementById('typingIndicator');
        const tempSlider = document.getElementById('temperature');
        const tempValue = document.getElementById('tempValue');
        const lengthSlider = document.getElementById('maxLength');
        const lengthValue = document.getElementById('lengthValue');
        
        let isWaiting = false;

        // Update slider values
        tempSlider.oninput = function() {
            tempValue.textContent = this.value;
        }
        
        lengthSlider.oninput = function() {
            lengthValue.textContent = this.value;
        }

        // Add message to chat
        function addMessage(content, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;
            
            messageDiv.innerHTML = `
                <div>
                    <div class="message-label">${isUser ? 'You' : 'YouAI'}</div>
                    <div class="message-bubble">${content}</div>
                </div>
            `;
            
            chatContainer.insertBefore(messageDiv, typingIndicator);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Set example prompt
        function setPrompt(text) {
            userInput.value = text;
            userInput.focus();
        }

        // Send message
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message || isWaiting) return;

            // Add user message
            addMessage(message, true);
            
            userInput.value = '';
            isWaiting = true;
            sendBtn.disabled = true;
            typingIndicator.style.display = 'block';

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: message,
                        temperature: parseFloat(tempSlider.value),
                        max_length: parseInt(lengthSlider.value)
                    })
                });

                const data = await response.json();
                
                if (data.status === 'success') {
                    addMessage(data.response, false);
                } else {
                    addMessage('Sorry, I encountered an error generating a response.', false);
                }
            } catch (error) {
                addMessage('Oops! Connection error. Is the server running?', false);
            } finally {
                typingIndicator.style.display = 'none';
                isWaiting = false;
                sendBtn.disabled = false;
                userInput.focus();
            }
        }

        // Event listeners
        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        // Make setPrompt globally available
        window.setPrompt = setPrompt;
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate text endpoint"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        temperature = data.get('temperature', 0.8)
        max_length = data.get('max_length', 100)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Generate response
        responses = youai_model.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9
        )
        
        return jsonify({
            'response': responses[0],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check model status"""
    return jsonify({
        'model_loaded': youai_model is not None,
        'status': 'ready' if youai_model else 'not_loaded'
    })


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YouAI Web Interface')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run the server on'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to run the server on'
    )
    
    args = parser.parse_args()
    
    # Load model
    global youai_model
    print("=" * 60)
    print("Loading YouAI model...")
    print("=" * 60)
    
    youai_model = YouAIInference(args.checkpoint)
    
    print("\n" + "=" * 60)
    print("YouAI Web Interface is ready! 🚀")
    print(f"Open your browser to: http://localhost:{args.port}")
    print("=" * 60 + "\n")
    
    # Start server
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
