
class Coder:
    def __init__(self, vocab: map):
        self.token_to_index = {token: i for i, token in enumerate(vocab)}
        self.index_to_token = {i: token for i, token in enumerate(vocab)}

    def encode(self, tokens: list) -> list:
        encoded = []
        for token in tokens:
            encoded.append(self.token_to_index.get(token, -1))
        return encoded

    def decode(self, tokens: list) -> list:
        decoded = []
        for token in tokens:
            decoded.append(self.index_to_token.get(token, -1))
        return decoded