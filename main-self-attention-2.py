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
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value
print(x_2)
print(W_query)
print(query_2)

