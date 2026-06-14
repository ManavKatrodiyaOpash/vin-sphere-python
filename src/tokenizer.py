from src.config import ALLOWED_CHARS, PAD_TOKEN, UNK_TOKEN

class VINTokenizer:
    """Custom VIN Tokenizer mapped to alphanumeric vocabulary."""
    def __init__(self):
        self.chars = ALLOWED_CHARS
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        
        self.char_to_id = {self.pad_token: 0, self.unk_token: 1}
        for i, char in enumerate(self.chars):
            self.char_to_id[char] = i + 2
            
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)

    def encode(self, vin: str, max_length: int = 17) -> list:
        tokens = []
        for char in vin[:max_length]:
            tokens.append(self.char_to_id.get(char, self.char_to_id[self.unk_token]))
        # Padding
        if len(tokens) < max_length:
            tokens += [self.char_to_id[self.pad_token]] * (max_length - len(tokens))
        return tokens

    def decode(self, token_ids: list) -> str:
        return "".join([self.id_to_char.get(tid, self.unk_token) for tid in token_ids])
