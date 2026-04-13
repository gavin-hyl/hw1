from collections.abc import Iterable
import json

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
  input_path: str,
  vocab_size: int,
  special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

  with open(input_path, encoding="utf-8") as f:
    text = f.read()

  # Split on special tokens so no merging occurs across them
  if special_tokens:
    special_pattern = "|".join(regex.escape(t) for t in special_tokens)
    chunks = regex.split(special_pattern, text)
  else:
    chunks = [text]

  # Count word frequencies
  word_freqs = {}
  for chunk in chunks:
    for match in regex.finditer(PAT, chunk):
      word = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
      word_freqs[word] = word_freqs.get(word, 0) + 1

  # Initialize pair frequencies and pair-to-words mapping
  pair_freqs = {}
  pair2words = {}
  for word, freq in word_freqs.items():
    for i in range(len(word) - 1):
      pair = (word[i], word[i + 1])
      pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
      if pair not in pair2words:
        pair2words[pair] = set()
      pair2words[pair].add(word)

  n_merges = vocab_size - 256 - len(special_tokens)
  merges = []
  for _ in range(n_merges):
    if not pair_freqs:
      break
    # get the most frequent pair, tiebreak on lexicographic order
    max_pair, _ = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))
    pair_freqs.pop(max_pair)
    new_token = max_pair[0] + max_pair[1]
    merges.append(max_pair)

    # get the words that are affected by the merge
    affected_words = pair2words.pop(max_pair, set())
    for word in list(affected_words):
      # Build new word by applying the merge
      new_word = []
      i = 0
      while i < len(word):
        if i < len(word) - 1 and word[i] == max_pair[0] and word[i + 1] == max_pair[1]:
          new_word.append(new_token)
          i += 2
        else:
          new_word.append(word[i])
          i += 1
      new_word = tuple(new_word)

      # Count pairs in old and new words
      old_counts = {}
      for j in range(len(word) - 1):
        p = (word[j], word[j + 1])
        old_counts[p] = old_counts.get(p, 0) + 1

      new_counts = {}
      for j in range(len(new_word) - 1):
        p = (new_word[j], new_word[j + 1])
        new_counts[p] = new_counts.get(p, 0) + 1

      freq = word_freqs[word]

      # Update pair frequencies and pair2words for all affected pairs
      for p in set(old_counts) | set(new_counts):
        if p == max_pair:
          continue
        old_c = old_counts.get(p, 0)
        new_c = new_counts.get(p, 0)
        delta = new_c - old_c

        if delta != 0:
          pair_freqs[p] = pair_freqs.get(p, 0) + delta * freq

        if pair_freqs.get(p, 0) <= 0:
          pair_freqs.pop(p, None)
          if p in pair2words:
            pair2words[p].discard(word)
            if not pair2words[p]:
              pair2words.pop(p)
          continue

        # Update word references in pair2words
        if old_c > 0 and p in pair2words:
          pair2words[p].discard(word)
        if new_c > 0:
          if p not in pair2words:
            pair2words[p] = set()
          pair2words[p].add(new_word)

      # Update the tokenized word
      word_freqs[new_word] = word_freqs.pop(word)

  # Build vocab
  vocab = {}
  token_id = 0
  # First add special tokens
  for token in special_tokens:
    vocab[token_id] = token.encode("utf-8")
    token_id += 1
  # Then add all 256 byte tokens
  for i in range(256):
    vocab[token_id] = bytes([i])
    token_id += 1
  # Then add merged tokens
  for pair in merges:
    vocab[token_id] = pair[0] + pair[1]
    token_id += 1

  return vocab, merges


def serialize_vocab(vocab: dict[int, bytes]) -> dict[str, str]:
  return {str(k): v.hex() for k, v in vocab.items()}

def deserialize_vocab(serialized_vocab: dict[str, str]) -> dict[int, bytes]:
  return {int(k): bytes.fromhex(v) for k, v in serialized_vocab.items()}

def serialize_merges(merges: list[tuple[bytes, bytes]]) -> list[str]:
  return [f"{a.hex()} {b.hex()}" for a, b in merges]

def deserialize_merges(serialized_merges: list[str]) -> list[tuple[bytes, bytes]]:
  merges = []
  for line in serialized_merges:
    a_hex, b_hex = line.strip().split()
    merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))
  return merges


class Tokenizer:
  """
  Given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text. Also supports special tokens.
  """

  def __init__(self,
               vocab: dict[int, bytes],
               merges: list[tuple[bytes, bytes]],
               special_tokens: list[str] | None = None):
    self.vocab = dict(vocab)
    self.merges = merges
    # bytes -> token_id reverse lookup
    self.bytes_to_id = {v: k for k, v in self.vocab.items()}
    # merge pair -> rank for O(1) lookup
    self.merge_rank = {pair: i for i, pair in enumerate(merges)}
    # Handle special tokens: sort longest-first for overlapping matches
    self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
    self.special_token_ids = {}
    for st in self.special_tokens:
      st_bytes = st.encode("utf-8")
      if st_bytes in self.bytes_to_id:
        self.special_token_ids[st] = self.bytes_to_id[st_bytes]
      else:
        new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        self.vocab[new_id] = st_bytes
        self.bytes_to_id[st_bytes] = new_id
        self.special_token_ids[st] = new_id

  @classmethod
  def from_files(cls,
                 vocab_filepath: str,
                 merges_filepath: str,
                 special_tokens: list[str] | None = None):
    with open(vocab_filepath) as f:
        vocab = deserialize_vocab(json.load(f))
    with open(merges_filepath) as f:
        merges = deserialize_merges(f.readlines())
    return cls(vocab, merges, special_tokens)

  def encode(self, text: str) -> list[int]:
    if not text:
      return []
    ids = []
    if self.special_tokens:
      special_pattern = "(" + "|".join(regex.escape(t) for t in self.special_tokens) + ")"
      parts = regex.split(special_pattern, text)
    else:
      parts = [text]

    for i, part in enumerate(parts):
      if not part:
        continue
      if i % 2 == 1:  # special token
        ids.append(self.special_token_ids[part])
      else:
        for match in regex.finditer(PAT, part):
          word = [bytes([b]) for b in match.group(0).encode("utf-8")]
          while len(word) >= 2:
            best_rank = float('inf')
            best_idx = -1
            for j in range(len(word) - 1):
              rank = self.merge_rank.get((word[j], word[j + 1]))
              if rank is not None and rank < best_rank:
                best_rank = rank
                best_idx = j
            if best_idx == -1:
              break
            merged = word[best_idx] + word[best_idx + 1]
            new_word = []
            j = 0
            while j < len(word):
              if j < len(word) - 1 and (word[j], word[j + 1]) == (word[best_idx], word[best_idx + 1]) and self.merge_rank.get((word[j], word[j + 1])) == best_rank:
                new_word.append(merged)
                j += 2
              else:
                new_word.append(word[j])
                j += 1
            word = new_word
          for token in word:
            ids.append(self.bytes_to_id[token])
    return ids

  def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
    for text in iterable:
      yield from self.encode(text)

  def decode(self, ids: list[int]) -> str:
    return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")