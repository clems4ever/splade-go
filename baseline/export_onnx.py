import torch
from sentence_transformers import SparseEncoder
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Model name
model_name = "naver/splade-cocondenser-ensembledistil"

# 1. Load encoder wrapper
encoder = SparseEncoder(model_name)

# 2. Grab underlying HF model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForMaskedLM.from_pretrained(model_name)
hf_model.eval()

# 3. Dummy text and inputs
dummy_text = "export test to onnx"
inputs = tokenizer(dummy_text, return_tensors="pt")

# 4. Export raw masked LM logits
torch.onnx.export(
    hf_model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "splade_raw.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    },
    opset_version=14,
    do_constant_folding=True,
)

print("Exported raw SPLADE model to splade_raw.onnx")

# ---------------------------------------------------------
# OPTIONAL: Export *pooled SPLADE embeddings* (batch, 30522)
# ---------------------------------------------------------

class SpladeWrapper(torch.nn.Module):
    """Wrap HF model with SPLADE pooling -> [batch, vocab_size]."""
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids, attention_mask):
        outputs = self.hf_model(input_ids=input_ids,
                                attention_mask=attention_mask).logits
        # SPLADE activation + pooling (log(1+relu))
        activation = torch.log1p(torch.relu(outputs))
        # Max pooling over sequence length
        sparse_vec = activation.max(dim=1).values
        return sparse_vec

splade_model = SpladeWrapper(hf_model).eval()

torch.onnx.export(
    splade_model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "splade_pooled.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["sparse_embedding"],  # directly the [batch, 30522] vec
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "sparse_embedding": {0: "batch"},
    },
    opset_version=14,
    do_constant_folding=True,
)

print("Exported pooled SPLADE model to splade_pooled.onnx")