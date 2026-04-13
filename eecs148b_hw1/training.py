import torch
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import datetime as dt
from eecs148b_hw1.transformer import TransformerLM
from eecs148b_hw1.constants import CONTEXT_LENGTH, N_LAYERS, D_MODEL, N_HEADS, TINYSTORIES_TRAIN_ENC, TINYSTORIES_VAL_ENC, MODEL_DIR, USE_LAYERNORM, USE_SIN_PE, VOCAB_SIZE

def cross_entropy_loss(logits, labels):
  logits = logits - logits.max(dim=-1, keepdim=True).values
  log_sum_exp = torch.exp(logits).sum(dim=-1).log()
  correct_logits = logits[torch.arange(logits.shape[0]), labels]
  return (log_sum_exp - correct_logits).mean()


def get_batch(x, batch_size, context_length, device='cpu'):
  random_idx = np.random.randint(0, x.shape[0] - context_length, size=batch_size)
  x_batch = torch.stack([torch.from_numpy(x[i:i+context_length].copy()) for i in random_idx], dim=0).long().to(device)
  y_batch = torch.stack([torch.from_numpy(x[i+1:i+context_length+1].copy()) for i in random_idx], dim=0).long().to(device)
  return x_batch, y_batch



def get_batch_deterministic(x, batch_size, context_length, start_idx, device='cpu'):
  indices = range(start_idx, start_idx + batch_size * context_length, context_length)
  x_batch = torch.stack([torch.from_numpy(x[i:i+context_length].copy()) for i in indices], dim=0).long().to(device)
  y_batch = torch.stack([torch.from_numpy(x[i+1:i+context_length+1].copy()) for i in indices], dim=0).long().to(device)
  return x_batch, y_batch


def evaluate_full(model, data, batch_size, context_length, device='cpu'):
  model.eval()
  total_loss = 0.0
  n_batches = 0
  with torch.no_grad():
    for start in range(0, len(data) - context_length - 1, batch_size * context_length):
      actual_bs = min(batch_size, (len(data) - context_length - 1 - start) // context_length)
      if actual_bs <= 0:
        break
      x_batch, y_batch = get_batch_deterministic(data, actual_bs, context_length, start, device)
      logits = model(x_batch)
      loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y_batch.view(-1).long())
      total_loss += loss.item()
      n_batches += 1
  return total_loss / n_batches


def find_latest_checkpoint(run_dir):
  """Find the checkpoint with the highest epoch number in run_dir."""
  import re
  checkpoints = list(run_dir.glob('model_epoch_*.pth'))
  if not checkpoints:
    return None, 0
  best = max(checkpoints, key=lambda p: int(re.search(r'model_epoch_(\d+)', p.stem).group(1)))
  start_epoch = int(re.search(r'model_epoch_(\d+)', best.stem).group(1))
  return best, start_epoch


def train(epochs: int,
          lr: float,
          batch_size: int,
          context_length: int,
          d_model: int,
          num_heads: int,
          d_ff: int,
          num_layers: int,
          log_interval: int = 10,
          n_val_batches: int = -1,
          tag: str = 'default',
          resume: str | None = None):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = TransformerLM(d_model, num_heads, d_ff, VOCAB_SIZE, context_length, num_layers, USE_LAYERNORM, USE_SIN_PE).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  start_epoch = 0
  if resume:
    from pathlib import Path
    run_dir = Path(resume)
    ckpt_path, start_epoch = find_latest_checkpoint(run_dir)
    if ckpt_path:
      model.load_state_dict(torch.load(ckpt_path, map_location=device))
      print(f'Resumed from {ckpt_path} (epoch {start_epoch})')
    else:
      print(f'No checkpoint found in {run_dir}, starting fresh')
  else:
    ymd_hm = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = MODEL_DIR / ymd_hm
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'tag.txt', 'w') as f:
      f.write(tag)
  writer = SummaryWriter(log_dir=str(run_dir))
  # Load data
  train_data = np.load(TINYSTORIES_TRAIN_ENC, mmap_mode='r')
  val_data = np.load(TINYSTORIES_VAL_ENC, mmap_mode='r')
  end_epoch = start_epoch + epochs
  for epoch in range(start_epoch, end_epoch):
    x_batch, y_batch = get_batch(train_data, batch_size, context_length, device)
    logits = model(x_batch)
    loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y_batch.view(-1).long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar('loss/train', loss.item(), epoch)
    if (epoch + 1) % log_interval == 0:
      model.eval()
      val_losses = []
      with torch.no_grad():
        # randomly sample validation batches and avg their losses
        if n_val_batches < 0:
          n_val_batches = (len(val_data) - context_length) // (batch_size * context_length)
        for _ in range(n_val_batches):
          vx, vy = get_batch(val_data, batch_size, context_length, device)
          val_logits = model(vx)
          val_losses.append(cross_entropy_loss(val_logits.view(-1, val_logits.size(-1)), vy.view(-1).long()).item())
      val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
      model.train()
      writer.add_scalar('loss/val', val_loss, epoch)
      print(f'Epoch {epoch+1}/{end_epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
      # save model checkpoint
      torch.save(model.state_dict(), run_dir / f'model_epoch_{epoch+1}.pth')
  writer.close()


def main():
  # parse command line arguments and call train()
  import argparse
  parser = argparse.ArgumentParser(description='Train a Transformer LM on TinyStories')
  parser.add_argument('--epochs', type=int, default=20000, help='Number of training epochs')
  parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
  parser.add_argument('--d_ff', type=int, default=4096, help='Dimension of feedforward network')
  parser.add_argument('--log_interval', type=int, default=100, help='Interval for logging training progress')
  parser.add_argument('--tag', type=str, required=True, help='Tag/description for this training run')
  parser.add_argument('--n_val_batches', type=int, default=20, help='Number of validation batches to evaluate at each log interval (set to -1 to use entire validation set)')
  parser.add_argument('--resume', type=str, default=None, help='Path to a run directory to resume training from (e.g. data/models/2026-04-05_15-48-25)')
  args = parser.parse_args()
  train(epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        d_ff=args.d_ff,
        num_layers=N_LAYERS,
        log_interval=args.log_interval,
        n_val_batches=args.n_val_batches,
        tag=args.tag,
        resume=args.resume)

if __name__ == '__main__':
  main()