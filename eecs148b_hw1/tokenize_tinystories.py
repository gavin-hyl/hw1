from eecs148b_hw1.bpe_tokenizer import Tokenizer
from eecs148b_hw1.constants import TINYSTORIES_TRAIN, OUTPUT_MERGES, OUTPUT_VOCAB, SPECIAL_TOKENS, TINYSTORIES_VAL, TINYSTORIES_TRAIN_ENC, TINYSTORIES_VAL_ENC
import random
import numpy as np

def main():
  tokenizer = Tokenizer.from_files(str(OUTPUT_VOCAB), str(OUTPUT_MERGES), SPECIAL_TOKENS)
  with open(TINYSTORIES_TRAIN) as f:
    text = f.read()
    
  # split the text and grab the first 10 stories
  stories = text.split("<|endoftext|>")
  stories = random.sample(stories, 10)
  encoded_total_length = 0
  raw_total_length = 0
  for story in stories:
    ids = tokenizer.encode(story)
    encoded_total_length += len(ids)
    raw_total_length += len(story.encode("utf-8"))
    
  print(f"bytes/token: {raw_total_length / encoded_total_length:.2f}")
  
  enc_train = tokenizer.encode_iterable(open(TINYSTORIES_TRAIN))
  np.savez_compressed(TINYSTORIES_TRAIN_ENC, data=np.array(list(enc_train), dtype=np.uint16))
  print(f"Encoded training set saved to {TINYSTORIES_TRAIN_ENC}")

  enc_val = tokenizer.encode_iterable(open(TINYSTORIES_VAL))
  np.savez_compressed(TINYSTORIES_VAL_ENC, data=np.array(list(enc_val), dtype=np.uint16))
  print(f"Encoded validation set saved to {TINYSTORIES_VAL_ENC}")
  

if __name__ == "__main__":
  main()