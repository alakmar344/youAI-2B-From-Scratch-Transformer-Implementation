"""YouAI Streaming Generation

Real-time token-by-token generation for interactive applications.

Features:
- Token-by-token streaming for real-time output
- Callback-based generation
- Async generator support
- Progress tracking
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Generator, AsyncGenerator, List
from transformers import GPT2Tokenizer
import asyncio


class StreamingGenerator:
    """Generate text with token-by-token streaming.
    
    Perfect for:
    - Chat applications
    - Real-time text display
    - Interactive terminals
    - Web interfaces
    
    Usage:
        from youai.streaming import StreamingGenerator
        
        generator = StreamingGenerator(model, tokenizer)
        
        # Method 1: Callback
        for token in generator.stream("Hello"):
            print(token, end="", flush=True)
        
        # Method 2: Generator
        for token in generator.generate_stream("Hello"):
            print(token, end="", flush=True)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: GPT2Tokenizer,
        device: str = "cuda",
    ):
        """
        Args:
            model: YouAIModel instance
            tokenizer: GPT2Tokenizer
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        
        self.model.to(self.device).eval()
    
    def stream(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_tokens: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Stream tokens one by one.
        
        Args:
            prompt: Input text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            stop_tokens: List of stop strings
            
        Yields:
            Generated tokens one at a time
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_text = ""
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Decode token
                token_text = self.tokenizer.decode(next_token[0])
                generated_text += token_text
                
                # Check stop tokens
                if stop_tokens:
                    should_stop = False
                    for stop in stop_tokens:
                        if stop in generated_text:
                            should_stop = True
                            break
                    if should_stop:
                        break
                
                yield token_text
                
                # Update input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    def generate_stream(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text with streaming and optional callback.
        
        Args:
            prompt: Input text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            callback: Function called with each token
            
        Returns:
            Complete generated text
        """
        full_text = ""
        
        for token in self.stream(prompt, max_length, temperature, top_k, top_p):
            full_text += token
            if callback:
                callback(token)
        
        return full_text
    
    async def async_stream(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        """Async generator for streaming tokens.
        
        Args:
            prompt: Input text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Yields:
            Generated tokens one at a time
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                token_text = self.tokenizer.decode(next_token[0])
                yield token_text
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                await asyncio.sleep(0)  # Yield control


class ChatSession:
    """Interactive chat session with streaming support.
    
    Maintains conversation history and supports streaming responses.
    
    Usage:
        from youai.streaming import ChatSession
        
        session = ChatSession(model, tokenizer)
        
        # Simple chat
        response = session.chat("Hello!")
        
        # Streaming chat
        for token in session.chat_stream("Hello!"):
            print(token, end="", flush=True)
        
        # Get history
        print(session.get_history())
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: GPT2Tokenizer,
        system_prompt: str = "You are a helpful AI assistant.",
        max_history: int = 10,
        device: str = "auto",
    ):
        """
        Args:
            model: YouAIModel instance
            tokenizer: GPT2Tokenizer
            system_prompt: System prompt for chat
            max_history: Maximum conversation turns to keep
            device: Device to use
        """
        self.generator = StreamingGenerator(model, tokenizer, device)
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: List[dict] = []
    
    def _build_prompt(self, user_message: str) -> str:
        """Build prompt with conversation history."""
        prompt = f"{self.system_prompt}\n\n"
        
        # Add history
        for turn in self.history[-self.max_history:]:
            prompt += f"Human: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
        
        prompt += f"Human: {user_message}\nAssistant:"
        return prompt
    
    def chat(
        self,
        message: str,
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Send a message and get a complete response.
        
        Args:
            message: User message
            max_length: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            Assistant response
        """
        prompt = self._build_prompt(message)
        
        response = self.generator.generate_stream(
            prompt,
            max_length=max_length,
            temperature=temperature,
            stop_tokens=["Human:", "\n\n"],
        )
        
        # Clean up response
        response = response.strip()
        if response.startswith("Assistant:"):
            response = response[len("Assistant:"):].strip()
        
        # Store in history
        self.history.append({
            "user": message,
            "assistant": response,
        })
        
        return response
    
    def chat_stream(
        self,
        message: str,
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> Generator[str, None, str]:
        """Send a message and stream the response.
        
        Args:
            message: User message
            max_length: Maximum response length
            temperature: Sampling temperature
            
        Yields:
            Response tokens one at a time
            
        Returns:
            Complete response (via StopIteration.value)
        """
        prompt = self._build_prompt(message)
        full_response = ""
        
        for token in self.generator.stream(
            prompt,
            max_length=max_length,
            temperature=temperature,
            stop_tokens=["Human:", "\n\n"],
        ):
            full_response += token
            yield token
        
        # Clean and store
        full_response = full_response.strip()
        if full_response.startswith("Assistant:"):
            full_response = full_response[len("Assistant:"):].strip()
        
        self.history.append({
            "user": message,
            "assistant": full_response,
        })
        
        return full_response
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
    
    def get_history(self) -> List[dict]:
        """Get conversation history."""
        return self.history.copy()
    
    def set_system_prompt(self, prompt: str):
        """Update system prompt."""
        self.system_prompt = prompt
