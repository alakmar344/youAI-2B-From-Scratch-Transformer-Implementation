"""YouAI Inference Module"""

import os
import json
from typing import List, Optional

import torch
from transformers import GPT2Tokenizer

from .model import YouAIModel
from .config import YouAIConfig


class YouAIInference:
    """Inference wrapper for YouAI model.
    
    Args:
        checkpoint_path: Path to model checkpoint directory
        device: Device to use ('cuda' or 'cpu')
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        
        config_path = os.path.join(checkpoint_path, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = YouAIConfig.from_dict(config_dict)
        
        self.model = YouAIModel(config)
        
        weights_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of responses to generate
            
        Returns:
            List of generated text strings
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output_sequences = []
            for _ in range(num_return_sequences):
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                output_sequences.append(output_ids)
        
        generated_texts = []
        for output_ids in output_sequences:
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def chat(
        self,
        message: str,
        history: Optional[List[dict]] = None,
        max_length: int = 150,
        temperature: float = 0.8,
    ) -> str:
        """Generate a chat response.
        
        Args:
            message: User message
            history: Conversation history (list of {'role': 'user'/'assistant', 'content': str})
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Assistant response text
        """
        prompt = ""
        if history:
            for msg in history[-4:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    prompt += f"Human: {content}\n"
                else:
                    prompt += f"YouAI: {content}\n"
        
        prompt += f"Human: {message}\nYouAI:"
        
        responses = self.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9
        )
        
        response = responses[0]
        
        if "YouAI:" in response:
            ai_response = response.split("YouAI:")[-1].strip()
            if "Human:" in ai_response:
                ai_response = ai_response.split("Human:")[0].strip()
        else:
            ai_response = response[len(prompt):].strip()
        
        return ai_response
