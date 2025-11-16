import torch

# 1) Simplified self-attention

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# 1.1) Exemple pour le token "journey".
query = inputs[1]                            #1
# Calcul des scores d'attention.
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # Multiplication scalaire entre x_i et la query.

print(attn_scores_2) 
#1 The second input token serves as the query.

# V1 Simplifiée.
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# V2 Simplifiée avec softmax.
# def softmax_naive(x):
#     return torch.exp(x) / torch.exp(x).sum(dim=0)

# attn_weights_2_naive = softmax_naive(attn_scores_2)
# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())

# V3 avec la fonction softmax de PyTorch.
# Calcule les poids d'attention.
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# Calcul du context vector.
query = inputs[1]         #1
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)

#1.2) Computing attention wieights for all input tokens

# 1.2.1) Multiplication de matrice pour les scores d'attention
# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)

# print(attn_scores)

# 1.2.2) Simplification de la multiplication de matrice. (plus rapide et plus concise)
attn_scores = inputs @ inputs.T
print(attn_scores)
print(attn_scores_2)
attn_scores_2 = attn_scores[:, 1]
print(attn_scores_2)

# 1.3) Calcul des poids d'attention avec softmax
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# 1.4) Calcul des context vectors
# On utilise simplement une multiplication de matrice.
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)



 