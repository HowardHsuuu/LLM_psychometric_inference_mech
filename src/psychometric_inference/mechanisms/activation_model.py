"""
Activation extraction from Llama 3.1 8B.

Provides hook-based extraction of post-MLP residual stream activations
at specified layers during forward passes. Supports:
  1. Prompt-only extraction (last token activation after reading persona)
  2. Response extraction (mean activation across generated tokens)

Usage:
    model = ActivationModel("meta-llama/Llama-3.1-8B-Instruct")
    
    # Extract last-token activation after reading a prompt
    act = model.extract_prompt_activation(system_prompt, user_prompt, layer=16)
    
    # Generate response and extract mean response activation
    act, text = model.extract_response_activation(system_prompt, user_prompt, layer=16)
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config import MODEL_ID, DEVICE, MAX_NEW_TOKENS, TEMPERATURE, N_LAYERS

logger = logging.getLogger(__name__)


class ActivationModel:
    """Wraps a HuggingFace causal LM with activation extraction capabilities."""

    def __init__(self, model_id: str = MODEL_ID, device: str = DEVICE, prompt_format: str = "raw"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {model_id} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device
        )
        self.model.eval()
        self.model_id = model_id
        self.device = device
        self.n_layers = self.model.config.num_hidden_layers
        self.prompt_format = prompt_format

        if prompt_format not in {"raw", "chat_template"}:
            raise ValueError(f"Unknown prompt_format: {prompt_format}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loaded: {model_id}, {self.n_layers} layers, device={device}")

        # Storage for hooked activations
        self._hooked_activations: Dict[int, object] = {}
        self._hooks = []

    def _build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Build prompt for activation extraction."""
        if self.prompt_format == "chat_template":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if system_prompt:
            return f"{system_prompt}\n\n{user_prompt}\n\nAnswer: "
        return f"{user_prompt}\n\nAnswer: "

    def _register_hooks(self, layers: List[int]):
        """Register forward hooks on post-MLP residual stream at specified layers.
        
        In Llama, each transformer layer is model.model.layers[i].
        The residual stream after layer i = input + attn_output + mlp_output,
        which is the output of model.model.layers[i].
        """
        self._clear_hooks()
        self._hooked_activations = {}

        for layer_idx in layers:
            layer_module = self.model.model.layers[layer_idx]

            def make_hook(idx):
                def hook_fn(module, input, output):
                    # output is a tuple; first element is the hidden state
                    # Shape: (batch=1, seq_len, hidden_dim)
                    if isinstance(output, tuple):
                        self._hooked_activations[idx] = output[0].detach()
                    else:
                        self._hooked_activations[idx] = output.detach()
                return hook_fn

            h = layer_module.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

    def _clear_hooks(self):
        """Remove all registered hooks and free GPU memory from cached activations."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        # Explicitly delete GPU tensors before clearing the dict
        for key in list(self._hooked_activations.keys()):
            del self._hooked_activations[key]
        self._hooked_activations = {}

    def extract_prompt_activation(
        self,
        system_prompt: str,
        user_prompt: str,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """Extract last-token activation after processing the full prompt.
        
        This captures the model's representation of the persona + question
        at the moment just before it would generate a response.
        
        Args:
            system_prompt: Persona system prompt (e.g., mega prompt with all 114 items)
            user_prompt: The probe question
            layers: Which layers to extract. Default: all target layers.
            
        Returns:
            Dict mapping layer_idx -> activation vector (hidden_dim,)
        """
        if layers is None:
            from .config import TARGET_LAYERS
            layers = TARGET_LAYERS

        full_prompt = self._build_chat_prompt(system_prompt, user_prompt)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        self._register_hooks(layers)
        import torch
        with torch.no_grad():
            self.model(**inputs)

        # Extract last token activation from each layer
        result = {}
        for layer_idx in layers:
            if layer_idx in self._hooked_activations:
                # Shape: (1, seq_len, hidden_dim) -> take last token -> (hidden_dim,)
                act = self._hooked_activations[layer_idx][0, -1, :].cpu().float().numpy()
                result[layer_idx] = act

        self._clear_hooks()
        return result

    def extract_response_activation(
        self,
        system_prompt: str,
        user_prompt: str,
        layers: Optional[List[int]] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Tuple[Dict[int, np.ndarray], str]:
        """Generate a response and extract mean activation across response tokens.
        
        This is the persona vectors approach: generate under a persona condition,
        then average the residual stream activations across all generated tokens.
        
        Strategy:
          1. Encode prompt (get prompt length)
          2. Generate response token-by-token
          3. After generation, do one more forward pass on the full sequence
             (prompt + response) with hooks to get all activations
          4. Extract response token positions and mean-pool
          
        Args:
            system_prompt: Persona system prompt
            user_prompt: Probe question  
            layers: Which layers to extract
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (>0 for diversity)
            
        Returns:
            (activations_dict, generated_text)
            activations_dict maps layer_idx -> mean response activation (hidden_dim,)
        """
        if layers is None:
            from .config import TARGET_LAYERS
            layers = TARGET_LAYERS

        full_prompt = self._build_chat_prompt(system_prompt, user_prompt)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Generate response
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-7),
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_seq = outputs[0]  # (total_len,)
        response_tokens = full_seq[prompt_len:]
        generated_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        if len(response_tokens) == 0:
            self._clear_hooks()
            return {layer: np.zeros(self.model.config.hidden_size) for layer in layers}, ""

        # Forward pass on full sequence to get activations at all positions
        self._register_hooks(layers)
        with torch.no_grad():
            self.model(full_seq.unsqueeze(0))

        # Extract mean activation over response token positions
        result = {}
        for layer_idx in layers:
            if layer_idx in self._hooked_activations:
                # Shape: (1, total_len, hidden_dim)
                # Take positions [prompt_len:] and mean-pool
                response_acts = self._hooked_activations[layer_idx][0, prompt_len:, :]
                mean_act = response_acts.mean(dim=0).cpu().float().numpy()
                result[layer_idx] = mean_act

        self._clear_hooks()
        return result, generated_text

    def extract_logprobs(
        self,
        system_prompt: str,
        item_prompt: str,
        options: List[str],
    ) -> Dict[str, float]:
        """Extract logprobs for response options (for behavioral validation).
        
        Same as the existing pipeline's logprob scoring.
        """
        full_prompt = self._build_chat_prompt(system_prompt, item_prompt)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        result = {}
        for opt in options:
            token_ids = self.tokenizer.encode(opt, add_special_tokens=False)
            if token_ids:
                result[opt] = log_probs[token_ids[0]].item()
        return result

    def extract_activation_and_logprobs(
        self,
        system_prompt: str,
        item_prompt: str,
        options: List[str],
        layers: Optional[List[int]] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, float]]:
        """Extract BOTH last-token activation AND logprobs in a single forward pass.
        
        This is for the amplification experiment: we need to see both
        what the model represents (activation) and what it outputs (logprob)
        for the same item under the same persona condition.
        
        Returns:
            (activations_dict, logprobs_dict)
        """
        if layers is None:
            from .config import TARGET_LAYERS
            layers = TARGET_LAYERS

        full_prompt = self._build_chat_prompt(system_prompt, item_prompt)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        self._register_hooks(layers)
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Activations: last token at each layer
        activations = {}
        for layer_idx in layers:
            if layer_idx in self._hooked_activations:
                act = self._hooked_activations[layer_idx][0, -1, :].cpu().float().numpy()
                activations[layer_idx] = act

        self._clear_hooks()

        # Logprobs
        logits = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        logprobs = {}
        for opt in options:
            token_ids = self.tokenizer.encode(opt, add_special_tokens=False)
            if token_ids:
                logprobs[opt] = log_probs[token_ids[0]].item()

        return activations, logprobs

    def cleanup(self):
        """Free GPU memory."""
        logger.info(f"Cleaning up {self.model_id}")
        self._clear_hooks()
        del self.model
        del self.tokenizer
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
