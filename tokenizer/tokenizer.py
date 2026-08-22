from collections import Counter

def get_pair_counts(words):
    pair_counter = Counter()

    for symbol, frequency in words.items():
        for i in range(len(symbol) - 1):
            pair = (symbol[i], symbol[i + 1])
            pair_counter[pair] += frequency

    return pair_counter


def merge_pair(words, pair):
    new_words = {}

    for symbols, frequency in words.items():
        new_symbols = []

        i = 0

        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
                continue
            new_symbols.append(symbols[i])
            i += 1

        new_words[tuple(new_symbols)] = frequency

    return new_words


def train_bpe(words, num_merges):
    merges = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(words)

        if not pair_counts: break

        best_pair = pair_counts.most_common(1)[0][0] # [("token", freq)] format

        words = merge_pair(words, best_pair)

        merges.append(best_pair)

    return words, merges

def build_vocab(words, merges):
    special_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>"
    ]

    vocab = list(special_tokens)

    for symbols in words:
        for symbol in symbols:
            if symbol not in vocab:
                vocab.append(symbol)

    for left, right in merges:
        merged = left + right

        if merged not in vocab:
            vocab.append(merged)

    if " " not in vocab:
        vocab.append(" ")

    token_to_id = {
        token: i
        for i, token in enumerate(vocab)
    }

    id_to_token = {
        i: token
        for token, i in enumerate(vocab)
    }

    return token_to_id, id_to_token

def encode_word(word, merges, token_to_id):
    symbols = list(word)

    for pair in merges:
        new_symbols = []
        i = 0

        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
                continue
            new_symbols.append(symbols[i])
            i += 1

        symbols = new_symbols

    token_ids = [
        token_to_id.get(symbol, token_to_id["<unk>"])
        for symbol in symbols
    ]

    return symbols, token_ids

def encode_text(text, merges, token_to_id):
    words = text.split(" ")

    token_ids = []

    for i, word in enumerate(words):
        _, word_ids = encode_word(word, merges, token_to_id)

        token_ids.extend(word_ids)

        if i < len(words) - 1:
            token_ids.append(token_to_id[" "])

    return token_ids

if __name__ == "__main__":
    words = {
        ("l", "o", "w"): 3,
        ("l", "o", "w", "e", "r"): 2,
        ("n", "e", "w", "e", "s", "t"): 1,
        ("w", "i", "d", "e", "s", "t"): 1,
    }

    new_words, merges = train_bpe(words, num_merges = 5)

    token_to_id, id_to_token = build_vocab(words, merges)

    encoded = encode_text(
        "lower lowest",
        merges,
        token_to_id
    )

    print(encoded)