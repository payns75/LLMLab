import torch
import torch.nn as nn

# 1) Simplified self-attention

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


# Calcul du score d'attention
attn_scores = inputs @ inputs.T

# Calcul du poids d'attention
attn_weights = torch.softmax(attn_scores, dim=-1)

# Vecteur de contexte
all_context_vecs = attn_weights @ inputs


 ##########################################

x_2 = inputs[1]     #1
d_in = inputs.shape[1]      #2
d_out = 2         #3

#1 The second input element
#2 The input embedding size, d=3
#3 The output embedding size, d_out=2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query 
# key_2 = x_2 @ W_key 
# value_2 = x_2 @ W_value
# print(x_2)
# print(W_query)
# print(query_2)

 
keys = inputs @ W_key 
values = inputs @ W_value
print("keys.shape:", keys)
print("values.shape:", values.shape)

 
attn_scores_2 = query_2 @ keys.T       #1
print(attn_scores_2)

d_k = keys.shape[-1]                     #2
attn_weights_2 = torch.softmax(attn_scores_2 / (d_k**0.5), dim=-1)
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

 ##########################################

 
from Backend.SelfAttentionV1Class import SelfAttention_v1
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
print("sa_v1.W_query", sa_v1.W_query)


from Backend.SelfAttentionV2Class import SelfAttention_v2
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
print(sa_v2.W_query.weight)

## Je ne comprends pas pourquoi on peut retrouver les mêmes résultats en prenant le poids du v2 et en le transposant dans v1 alors qu'on a un seed différent.

W_query = nn.Linear(d_in, d_out, bias=False)
print("W_query.weight", W_query.weight)
print("W_query", W_query)

#############################################################
 
queries = sa_v2.W_query(inputs)     #1
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

masked_simple = attn_weights*mask_simple
print(masked_simple)

## TODO: Quelle est la différence entre * et @ dans ce contexte ?

## Premièr version de la normalisation
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

## Deuxième version de la normalisation avec softmax
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# 3.5.2 Masking additional attention weights with dropout
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)    #1
example = torch.ones(6, 6)      #2
print(dropout(example))
#1 We choose a dropout rate of 50%.
#2 Here, we create a matrix of 1s.
 
torch.manual_seed(123)
print(dropout(attn_weights))

## All together now!
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)                #1
#1 Two inputs with six tokens each; each token has embedding dimension 3.

from Backend.CausalAttentionClass import CausalAttention
torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)

## 3.6.1 Stacking multiple single-head attention layers

