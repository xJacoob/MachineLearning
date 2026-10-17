"""
Hugging Face basics - practice instead of reading (LLM Course, chapters 1-2).

Setup (once, in the MachineLearning project):
    uv add transformers

Models are downloaded on first run (a few hundred MB, cached afterwards).

How to work with this file:
- Go section by section. Read the CONCEPT comment, then fill in every TODO.
- Each section ends with a CHECK that tells you whether you got it right.
- Comment out later sections (or put `exit()` after the current one) while you work.
- Use the docs if you need: https://huggingface.co/docs/transformers
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BertTokenizer

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

sentences = [
    "I absolutely loved this course, it was great!",
    "This was boring.",
]

# =============================================================================
# SECTION 1 - The pipeline: what we want to reproduce by hand
# =============================================================================
# CONCEPT
# pipeline() hides three steps:
#   1. tokenizer  : text -> token ids (numbers), like encode() in Karpathy's GPT
#   2. model      : token ids -> logits (one score per class)
#   3. postprocess: logits -> probabilities (softmax) -> label
# In the rest of this file you will do these three steps yourself.

classifier = pipeline("text-classification", model=MODEL_NAME)
pipeline_results = classifier(sentences)
print("SECTION 1 - pipeline:", pipeline_results)
# CHECK: you see a list of dicts like {'label': 'POSITIVE', 'score': 0.99...}


# =============================================================================
# SECTION 2 - Tokenizer on a single sentence
# =============================================================================
# CONCEPT
# The tokenizer splits text into subword tokens (WordPiece for BERT) and maps
# each token to an id from its vocabulary. It also adds special tokens:
# [CLS] at the start and [SEP] at the end.

tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(type(tokenizer))

# TODO 2a: split sentences[0] into token strings (method: tokenize)
tokens = tokenizer.tokenize(sentences[0])
# TODO 2b: turn those token strings into ids (method: convert_tokens_to_ids)
ids = tokenizer.convert_tokens_to_ids(tokens)
# TODO 2c: call the tokenizer directly on sentences[0] (tokenizer(text))
#          and take its "input_ids"
encoded_ids = tokenizer(sentences[0])['input_ids']
# TODO 2d: decode encoded_ids back to text (method: decode)
decoded = tokenizer.decode(encoded_ids)

print("\nSECTION 2")
print("tokens     :", tokens)
print("ids        :", ids)
print("encoded_ids:", encoded_ids)
print("decoded    :", decoded)
# CHECK: encoded_ids is ids with one extra id at the start (101 = [CLS])
#        and one at the end (102 = [SEP]); decoded contains "[CLS]" and "[SEP]".
# QUESTION: why is "loved" or "absolutely" a single token, while a rare word
#           would be split into pieces starting with "##"?


# =============================================================================
# SECTION 3 - A batch of sentences: padding, truncation, attention_mask
# =============================================================================
# CONCEPT
# A model takes a tensor, and a tensor must be rectangular. Sentences have
# different lengths, so shorter ones are PADDED with [PAD] (id 0).
# The attention_mask says which positions are real tokens (1) and which are
# padding (0), so attention ignores the padding - similar to the causal mask
# in Karpathy's Head, but here it hides padding, not the future.
# Truncation cuts sentences that are longer than max_length.

# TODO 3: tokenize the whole `sentences` list at once with padding=True,
#         truncation=True and return_tensors="pt"
batch = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

print("\nSECTION 3")
print("input_ids shape:", batch["input_ids"].shape)
print(batch["input_ids"])
print(batch["attention_mask"])
# CHECK: shape is (2, T); the second (shorter) row ends with zeros in
#        input_ids AND zeros in attention_mask at the same positions.
# EXTRA: tokenize again with max_length=5 and see what truncation does.


# =============================================================================
# SECTION 4 - Model: from ids to logits to labels (= the pipeline, by hand)
# =============================================================================
# CONCEPT
# AutoModelForSequenceClassification = Transformer body + a classification head.
# It returns logits of shape (batch, num_classes). Softmax turns them into
# probabilities. model.config.id2label maps class index -> label name.

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

with torch.no_grad():
    # TODO 4a: run the model on the batch (hint: unpack the dict with **)
    outputs = model(**batch)

logits = outputs.logits
# TODO 4b: probabilities with softmax over the class dimension
probs = logits.softmax(dim=-1)
# TODO 4c: index of the most probable class for each sentence (argmax)
predicted_ids = torch.argmax(probs, dim=-1)
# TODO 4d: list of label names using model.config.id2label
labels = [model.config.id2label[int(label)] for label in predicted_ids]

print("\nSECTION 4")
print("logits:", logits)
print("probs :", probs)
print("labels:", labels)
# CHECK: labels and the highest probabilities match pipeline_results from
#        SECTION 1 (scores equal up to rounding).
for i, result in enumerate(pipeline_results):
    assert labels[i] == result["label"], f"Sentence {i}: label differs from pipeline"
    assert abs(probs[i].max().item() - result["score"]) < 1e-3, f"Sentence {i}: score differs"
print("SECTION 4 PASSED - you reproduced the pipeline by hand!")

# =============================================================================
# SECTION 5 - The body without a head: hidden states
# =============================================================================
# CONCEPT
# AutoModel loads only the Transformer body (no classification head).
# Its output last_hidden_state has shape (B, T, C) - exactly Karpathy's
# (B, T, C): one vector of size C (here 768 for DistilBERT) per token.
# The classification model from SECTION 4 uses the vector of the [CLS] token
# (position 0) to decide the class.

# TODO 5a: load the body with AutoModel.from_pretrained(MODEL_NAME)
body = AutoModel.from_pretrained(MODEL_NAME)
with torch.no_grad():
    # TODO 5b: run it on the batch
    body_out = body(**batch)

hidden = body_out.last_hidden_state
print("\nSECTION 5")
print("last_hidden_state shape:", hidden.shape)
# CHECK: shape is (2, T, 768) with the same T as in SECTION 3.
# TODO 5c: print the shape of the [CLS] vectors for both sentences
#          (hint: take position 0 on the T dimension) -> expected (2, 768)
print(body_out.last_hidden_state[:, 0].shape)


# =============================================================================
# SECTION 6 - Tokenizers compared: BPE (GPT-2) vs WordPiece (BERT)
# =============================================================================
# CONCEPT
# Karpathy's GPT used one character = one token (vocab 65).
# Real models use SUBWORDS: frequent words are one token, rare words are split
# into frequent pieces. BPE (GPT-2) builds its vocabulary by repeatedly merging
# the most frequent pair of symbols ("t"+"h" -> "th", "th"+"e" -> "the").
# WordPiece (BERT) is very similar. Different conventions:
#   BERT marks a continuation with "##"   (understand, ##able)
#   GPT-2 marks a word START with "Ġ"     (Ġunder, stand, able) - Ġ means "space before"

gpt2_tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
bert_tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

text = "Tokenization is surprisingly understandable, unbelievably!"

# TODO 6a: tokenize `text` with both tokenizers
gpt2_tokens = gpt2_tok.tokenize(text)
bert_tokens = bert_tok.tokenize(text)
# TODO 6b: vocabulary sizes of both tokenizers (attribute: vocab_size)
gpt2_vocab = gpt2_tok.vocab_size
bert_vocab = bert_tok.vocab_size

print("\nSECTION 6")
print("GPT-2:", gpt2_tokens, "| vocab:", gpt2_vocab)
print("BERT :", bert_tokens, "| vocab:", bert_vocab)
# CHECK: GPT-2 vocab ~50k, BERT ~30k (vs 65 in your character-level GPT).
# QUESTIONS:
#   1. Which tokenizer keeps capital letters and which lowercases? Why
#      ("uncased" in the model name)?
#   2. Why are subwords a good middle ground between characters (vocab 65,
#      very long sequences) and whole words (huge vocab, unknown words)?

# =============================================================================
# SECTION 7 - Model families (chapter 1 in one exercise)
# =============================================================================
# CONCEPT
#   encoder only    (BERT, DistilBERT): sees the WHOLE sentence in both
#                    directions -> understanding tasks: classification, NER
#   decoder only    (GPT): causal mask, sees only the past -> generating text
#                    (this is what you built with Karpathy)
#   encoder-decoder (T5, BART): encoder reads the input, decoder generates
#                    the output -> translation, summarization
#
# TODO 7: write your answers as comments below:
#   a) Which family for: sentiment of a review? summarizing an article?
#      continuing a story? translating PL -> EN?
#   b) Why is BERT, not GPT, the natural choice for Portfolio Project #2
#      (text classification)? Use the words "mask" and "both directions".