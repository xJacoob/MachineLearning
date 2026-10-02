import numpy as np
from wandb.apis.public import query_generator

hash_map = {}

with open("glove.6B.100d.txt", 'r', encoding='utf-8') as f:
    counter = 0
    while counter < 40000:
        line = f.readline().split()
        hash_map[line[0]] = np.array(line[1:], dtype=np.float32)
        counter += 1

def cosine_similarity(vec1, vec2):
    numerator = vec1.dot(vec2)
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return numerator / denominator

res1 = cosine_similarity(hash_map['king'], hash_map['queen'])
res2 = cosine_similarity(hash_map['football'], hash_map['soccer'])
res3 = cosine_similarity(hash_map['tokyo'], hash_map['japan'])
res4 = cosine_similarity(hash_map['cold'], hash_map['hot'])
res5 = cosine_similarity(hash_map['bike'], hash_map['chair'])

print(res1)
print(res2)
print(res3)
print(res4)
print(f"{res5}\n")

words_list = list(hash_map.keys())
matrix = np.array(list(hash_map.values()))
lengths= np.linalg.norm(matrix, axis=1, keepdims=True)
matrix_norm = matrix / lengths

def words_similarity(word, k):
    vec_norm = matrix_norm[words_list.index(word)]
    result = matrix_norm @ vec_norm
    result_list = np.argsort(result.flatten())[-(k+1):][::-1]
    similar_words = [words_list[i] for i in result_list]
    return similar_words

result = words_similarity("king", 10)
print(result[1:])

result = words_similarity("bank", 10)
print(result[1:])

king = matrix_norm[words_list.index('king')]
man = matrix_norm[words_list.index('man')]
woman = matrix_norm[words_list.index('woman')]

calc = king - man + woman
multiply = matrix_norm @ calc
result_list = np.argsort(multiply.flatten())[-(10+3):][::-1]
similar_words = [words_list[i] for i in result_list]
print(similar_words)
