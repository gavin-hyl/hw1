import torch
import numpy as np
from eecs148b_hw1.transformer import TransformerLM, softmax
from eecs148b_hw1.training import evaluate_full
from eecs148b_hw1.constants import CONTEXT_LENGTH, N_LAYERS, D_MODEL, N_HEADS, TINYSTORIES_VAL_ENC, MODEL_DIR, USE_LAYERNORM, USE_SIN_PE, VOCAB_SIZE, OUTPUT_VOCAB, OUTPUT_MERGES, SPECIAL_TOKENS
from eecs148b_hw1.bpe_tokenizer import Tokenizer

MODEL_PATH = MODEL_DIR / "2026-04-04_16-54-52" / "model_epoch_7400.pth"
D_FF = 4096
        
def generate(model: TransformerLM,
             context: np.ndarray,
             max_length: int,
             eot_token: int,
             nucleus_p: float = 1.0,
             temperature: float = 0.8) -> np.ndarray:
  device = next(model.parameters()).device
  model.eval()
  generated = torch.from_numpy(context).long().to(device)
  with torch.no_grad():
    for _ in range(max_length - len(context)):
      input_seq = generated[-CONTEXT_LENGTH:]
      logits = model(input_seq.unsqueeze(0)).squeeze(0)[-1] / (temperature + 1e-8)
      probs = softmax(logits)
      # nucleus sampling
      probs_sorted, indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
      i_needed = torch.searchsorted(cumulative_probs, nucleus_p)
      probs[indices[i_needed+1:]] = 0
      probs = probs / probs.sum()
      next_token = torch.multinomial(probs, num_samples=1)
      generated = torch.cat([generated, next_token])
      if next_token.item() == eot_token:
        break
  return generated.cpu().numpy()


def generate_text(model: TransformerLM,
                  tokenizer: Tokenizer,
                  eot_token: int,
                  prompt: str = '<|endoftext|>',
                  max_length: int=CONTEXT_LENGTH,
                  nucleus_p: float = 1.0,
                  temperature: float = 0.8) -> str:
  context = np.array(tokenizer.encode(prompt), dtype=np.uint16)
  generated_tokens = generate(model, context, max_length, eot_token, nucleus_p, temperature)
  return tokenizer.decode(generated_tokens[:].tolist())


def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = TransformerLM(D_MODEL, N_HEADS, D_FF, VOCAB_SIZE, CONTEXT_LENGTH, N_LAYERS, USE_LAYERNORM, USE_SIN_PE).to(device)
  model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

  val_data = np.load(TINYSTORIES_VAL_ENC, mmap_mode='r')
  print(f'Val data: {len(val_data)} tokens')

  val_loss = evaluate_full(model, val_data, batch_size=64, context_length=CONTEXT_LENGTH, device=device)
  print(f'Full val loss: {val_loss:.4f}')
  print(f'Val perplexity: {np.exp(val_loss):.2f}')
  
  # Generate samples
  tokenizer = Tokenizer.from_files(str(OUTPUT_VOCAB), str(OUTPUT_MERGES), SPECIAL_TOKENS)
  EOT_TOKEN = tokenizer.special_token_ids['<|endoftext|>']
  
  param_pairs = [
    (0.0, 1.0),
    (0.5, 1.0),
    (10.0, 1.0),
    (0.5, 0.5),
    (0.5, 0.01)
  ]
  
  for temperature, nucleus_p in param_pairs:
    print(f'Tau={temperature:.2f}, nucleus_p={nucleus_p:.2f}:')
    txt = generate_text(model, tokenizer, eot_token=EOT_TOKEN, temperature=temperature, nucleus_p=nucleus_p)  
    print(txt)
  
if __name__ == '__main__':
  main()
