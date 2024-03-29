import re
from collections import Counter


class SubwordTokenizer:
    def __init__(self, file_path, vocab_size):
        self.vocab_size = vocab_size
        self.subwords = self.learn_subwords(file_path)
        self.check_vocab_size()

    def learn_subwords(self, file_path):
        # Read text from file and preprocess
        text = self.read_text(file_path)

        # Initialize subwords with individual characters
        subwords = Counter(text)

        # Merge subwords based on frequency until vocab_size is reached
        while len(subwords) < self.vocab_size:
            most_common = subwords.most_common(1)[0][0]
            subword = re.escape(most_common)  # Escape special characters
            subwords.pop(most_common)
            for word in subwords:
                merged_word = "".join([most_common, word])
                subwords[merged_word] = subwords[word]
                subwords.pop(word)
        return subwords.keys()

    def check_vocab_size(self):
        if len(self.subwords) > self.vocab_size:
            raise ValueError(
                f"Provided vocab_size ({self.vocab_size}) is too small. Increase vocab_size to accommodate the learned vocabulary of {len(self.subwords)} subwords."
            )

    def read_text(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().lower()
        return text

    def tokenize(self, text):
        # Tokenize text into subwords
        tokens = []
        while text:
            found_subword = False
            for subword in self.subwords:
                if text.startswith(subword):
                    tokens.append(subword)
                    text = text[len(subword) :]
                    found_subword = True
                    break
            if not found_subword:
                tokens.append(text[0])  # Handle single characters
                text = text[1:]
        return tokens
