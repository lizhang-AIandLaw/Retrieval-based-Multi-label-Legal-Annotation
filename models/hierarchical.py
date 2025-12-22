import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class HierarchicalClassifier(nn.Module):
    def __init__(self, base_model_name, num_labels, num_layers=2, nhead=8, dim_feedforward=2048, dropout=0.1, trust_remote_code=True, attn_implementation=None):
        super().__init__()
        
        # Load base model
        kwargs = {"trust_remote_code": trust_remote_code}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
            
        self.base_model = AutoModel.from_pretrained(base_model_name, **kwargs)
        
        # Apply LoRA to base model if configured (passed via kwargs/external config? 
        # Ideally we should pass lora config to init)
        # For simplicity in this project structure, we check if 'peft' is available and apply it here if requested
        # But ModelConfig is not passed here. 
        # We will assume if the user wants LoRA for hierarchical, they might need to handle it outside or pass a config object.
        # Let's keep it simple: This class focuses on architecture. 
        # If LoRA is needed, it's best applied to self.base_model AFTER init in train.py
        
        self.config = self.base_model.config
        
        # Determine hidden size
        self.hidden_size = getattr(self.config, "hidden_size", getattr(self.config, "d_model", 768))
        
        # Freezing base model is optional, but often done in hierarchical to save memory. 
        # Here we keep it trainable as per typical fine-tuning unless specified otherwise.
        
        # Second level Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # input_ids: (batch_size, num_chunks, chunk_seq_len)
        batch_size, num_chunks, chunk_seq_len = input_ids.shape
        
        # Flatten for base model
        # (batch_size * num_chunks, chunk_seq_len)
        flat_input_ids = input_ids.view(-1, chunk_seq_len)
        flat_attention_mask = attention_mask.view(-1, chunk_seq_len) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(-1, chunk_seq_len) if token_type_ids is not None else None
        
        # Pass through base model
        outputs = self.base_model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids if "token_type_ids" in self.base_model.forward.__code__.co_varnames else None
        )
        
        # Extract embeddings
        # For BERT: use pooler_output or last_hidden_state[:, 0] (CLS)
        # For Qwen: use last_hidden_state and find last non-padding token or just last token if left-padded? 
        # Qwen is causal, so last token is usually the representation.
        # However, Qwen-Embedding might be used differently.
        # Let's try to support both generally.
        
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            # BERT-like
            chunk_embeddings = outputs.pooler_output
        else:
            # GPT/Qwen-like (No pooler)
            # Use the last token's embedding
            # Assuming right padding, we need to find the last real token.
            # But usually Qwen/GPT uses left padding for generation, but here we might stick to standard.
            # If we assume 'attention_mask' is 1 for real tokens and 0 for padding.
            hidden_states = outputs.last_hidden_state
            if flat_attention_mask is not None:
                # Find the index of the last token
                # sum(mask) - 1 gives index of last 1
                last_token_indices = flat_attention_mask.sum(1) - 1
                # Clamp to 0 just in case of empty sequence (shouldn't happen)
                last_token_indices = last_token_indices.clamp(min=0)
                
                # Gather
                # hidden_states: (B*N, L, D)
                # indices: (B*N)
                chunk_embeddings = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
            else:
                # Assume last token
                chunk_embeddings = hidden_states[:, -1, :]
        
        # Reshape back to (batch_size, num_chunks, hidden_size)
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, self.hidden_size)
        
        # Pass through second level transformer
        # We might want a mask for chunks that are purely padding (if any)
        # For simplicity, we assume all chunks provided are valid or padded chunks don't hurt much if processed.
        # But ideally we should mask out "empty" chunks if original doc had fewer than num_chunks.
        # We'll rely on the model learning to ignore padding chunks if we don't provide explicit chunk_mask.
        # (Constructing chunk_mask requires knowing which chunks were padding in data processing)
        
        encoded_chunks = self.transformer_encoder(chunk_embeddings)
        
        # Pool chunks to get document representation
        # Mean pooling
        doc_embedding = encoded_chunks.mean(dim=1)
        
        # Classify
        logits = self.classifier(doc_embedding)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            # Ensure labels are float
            if labels.dtype != torch.float32 and labels.dtype != torch.float16 and labels.dtype != torch.bfloat16:
                labels = labels.float()
            loss = loss_fct(logits, labels)
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.
        """
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def save_pretrained(self, save_directory):
        # Custom save to ensure base model + new layers are saved
        import os
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save the state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Save base model config (for loading base model type later)
        self.config.save_pretrained(save_directory)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # This is a simplified loader. 
        # For a real scenario, we'd handle loading the base model from config + loading state dict.
        # Since we are training from scratch (hierarchical structure), we mainly use this to init.
        # If loading a trained hierarchical model:
        
        # Check if it is a local directory with our custom binary
        import os
        if os.path.isdir(pretrained_model_name_or_path) and os.path.exists(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")):
            # Load config to get base_model_name
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            # We need 'base_model_name' and 'num_labels' from somewhere.
            # For now, let's assume kwargs provides them or we infer.
            # This part is tricky without a custom config class. 
            # To keep it simple for this project, we will instantiate the class normally and then load_state_dict.
            pass 
        
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

