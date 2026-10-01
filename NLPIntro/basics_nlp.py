from transformers import pipeline

classifier = pipeline("sentiment-analysis")
tokenizer = classifier.tokenizer

sentence_list = [
    "It's a great day",
    "I think he is bad at football",
    "I suppose this series is good, because the first part was quite interesting",
    "This exam was difficult, I don't think I will pass it",
    "Japan is such an amazing country, I have to visit it again"
]

results = classifier(sentence_list)

for sentence, result in zip(sentence_list, results):
    print(f"{sentence}: {result}")
    token_list = tokenizer.tokenize(sentence)
    print(token_list)
    print(tokenizer.convert_tokens_to_ids(token_list))
    print(tokenizer.encode(token_list))
    print()

print(f"Japan is such an amazing country, I have to visit it again {tokenizer.encode("Japan is such an amazing country, I have to visit it again")}")

tokens = tokenizer.tokenize("Understandable")
print(f"Understandable: {tokens}")
print(tokenizer.convert_tokens_to_ids(tokens))
print(f"Understandable: {tokenizer.encode("Understandable")}")

