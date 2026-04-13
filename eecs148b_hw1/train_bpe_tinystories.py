"""Train a BPE tokenizer on TinyStories and serialize vocab/merges to disk."""

import json
import time

from eecs148b_hw1.constants import TINYSTORIES_TRAIN, OUTPUT_MERGES, OUTPUT_VOCAB, SPECIAL_TOKENS, VOCAB_SIZE
from eecs148b_hw1.bpe_tokenizer import serialize_merges, serialize_vocab, train_bpe


def main():
    print(f"Training BPE tokenizer on {TINYSTORIES_TRAIN}")
    print(f"  vocab_size={VOCAB_SIZE}, special_tokens={SPECIAL_TOKENS}")

    start = time.time()
    vocab, merges = train_bpe(str(TINYSTORIES_TRAIN), VOCAB_SIZE, SPECIAL_TOKENS)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")

    # Serialize vocab as JSON: {token_id: hex-encoded bytes}
    with open(OUTPUT_VOCAB, "w") as f:
        json.dump(serialize_vocab(vocab), f, indent=2)
    print(f"Vocab written to {OUTPUT_VOCAB} ({len(vocab)} entries)")

    # Serialize merges as text: one merge per line, hex-encoded pair
    with open(OUTPUT_MERGES, "w") as f:
        for a, b in serialize_merges(merges):
            f.write(f"{a} {b}\n")
    print(f"Merges written to {OUTPUT_MERGES} ({len(merges)} entries)")

    # Inspect
    longest_token = max(vocab.values(), key=len)
    print(f"\nLongest token in vocab: {longest_token} ({len(longest_token)} bytes)")
    print(f"  Decoded: {longest_token.decode('utf-8')!r}")


if __name__ == "__main__":
    main()
