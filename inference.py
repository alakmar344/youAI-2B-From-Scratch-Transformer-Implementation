"""
Inference Script for YouAI
Use your trained model to generate text
"""

import torch
from transformers import GPT2Tokenizer
from model_architecture import YouAIModel, YouAIConfig
import json
import os


class YouAIInference:
    """Inference wrapper for YouAI model"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load config
        config_path = os.path.join(checkpoint_path, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config object
        config = YouAIConfig(**config_dict)
        
        # Create model
        print("Loading model...")
        self.model = YouAIModel(config)
        
        # Load weights
        weights_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
    
    def generate(
        self,
        prompt,
        max_length=100,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        num_return_sequences=1
    ):
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of responses to generate
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
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
        
        # Decode
        generated_texts = []
        for output_ids in output_sequences:
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def chat(self):
        """Interactive chat mode"""
        print("\n" + "=" * 60)
        print("YouAI Chat Interface")
        print("Type 'quit' to exit, 'clear' to clear history")
        print("=" * 60 + "\n")
        
        conversation_history = ""
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = ""
                print("Conversation cleared!")
                continue
            
            if not user_input:
                continue
            
            # Build prompt with history
            prompt = conversation_history + f"Human: {user_input}\nYouAI:"
            
            # Generate response
            responses = self.generate(
                prompt,
                max_length=150,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            response = responses[0]
            
            # Extract just the AI's response
            if "YouAI:" in response:
                ai_response = response.split("YouAI:")[-1].strip()
                if "Human:" in ai_response:
                    ai_response = ai_response.split("Human:")[0].strip()
            else:
                ai_response = response[len(prompt):].strip()
            
            print(f"YouAI: {ai_response}\n")
            
            # Update conversation history
            conversation_history += f"Human: {user_input}\nYouAI: {ai_response}\n"
            
            # Keep only last 4 exchanges to avoid context overflow
            lines = conversation_history.split('\n')
            if len(lines) > 8:
                conversation_history = '\n'.join(lines[-8:])


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YouAI Inference')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['chat', 'generate'],
        default='chat',
        help='Inference mode'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='',
        help='Prompt for generation mode'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Nucleus sampling threshold'
    )
    parser.add_argument(
        '--num_sequences',
        type=int,
        default=1,
        help='Number of sequences to generate'
    )
    
    args = parser.parse_args()
    
    # Load model
    youai = YouAIInference(args.checkpoint)
    
    if args.mode == 'chat':
        # Interactive chat
        youai.chat()
        
    else:
        # Single generation
        if not args.prompt:
            print("Error: --prompt required for generate mode")
            return
        
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...\n")
        
        responses = youai.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences
        )
        
        for i, response in enumerate(responses, 1):
            print(f"Response {i}:")
            print(response)
            print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
