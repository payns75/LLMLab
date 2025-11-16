# from importlib.metadata import version
import tiktoken
# print("tiktoken version:", version("tiktoken"))

# --- Byte pair encoding --- #
# Create a tokenizer for the GPT-2 model
# It uses byte pair encoding (BPE)
# Documentation: Ici, on démontre l'utilisation de la bibliothèque tiktoken pour obtenir un encodage BPE. L'idée ici est toute simple:
# - À partir d'un dictionnaire de mots, chaque mot est mappé à un entier unique.
# - Pour découper le/les texte(s), on split chacun des mots et on attribue un entier.
# - Pour les mots inconnus, on découpe le mot en sous-mots connues (allant jusqu'au caractère).
# - On ajoute ici une opion endoftext pour marquer la fin du texte. 
# - La librairie tiktoken contient déjà une liste de mots connus pour le modèle GPT-2. (exemple ici)

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace. sadfj . skdjfla k sdalkfj asdl;fjei kldsajf oien ef dkfja ;ldf ;aakjeiofjoiasdjf lkasdjf aosie jifjopaisdj fdfnh k."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# --- Data sampling --- #
# Le but ici va être de créer les pairs de tenseurs input-target pour l'entraînement du modèle LLM.
# On utilise une fenêtre glissante (sliding window) pour créer des séquences de longueur fixe (max_length).
# Chaque séquence d'entrée (input) est décalée d'un token par rapport à la séquence cible (target).
# On utilise un stride pour définir le décalage de la fenêtre glissante.
# TODO: Revenir dessus pour comprendre l'option shuffle.
# TODO: Revenir dessus pour comprendre le besoin de créer l'input et le target pour entrainer le modèle.

from Backend.GPTDatasetV1Class import create_dataloader_v1

dataloader = create_dataloader_v1(
    text, batch_size=8, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) 
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# --- Creating token embeddings --- #
# Cette étape consiste à convertir les tokens Ids en vecteurs denses (embeddings).
# La première chose à faire est d'intialiser des poids pour les embeddings de manière aléatoire.

# Par exemple, imaginons que nous avons la liste suivante de token IDs: 2, 3, 5, 1
# On imagine ensuite qui enous avons seulement 6 tokens dans notre vocabulaire (0 à 5).
# Enfin, on imagine que la dimension des embeddings est de 3.
# TODO: Essayer de mieux comprendre les dimensions.
import torch

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123) # Pour reproductibilité, du faux aléatoire.
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
vocab_weight = embedding_layer.weight # Les poids des embeddings qui correspondent à l'ensemble des tokens du vocabulaire.
print(vocab_weight)
print(embedding_layer(input_ids)) # Les embeddings pour les token IDs donnés. C'est tout simplement une opération de lookup (recherhe dans vocab_weight en fonction de l'id du token).

# -- Encoding word positions --- #
# Les modèles de type Transformer n'ont pas de notion de séquence temporelle.
# Pour intégrer cette information, on ajoute des embeddings de position aux embeddings de tokens.
# TODO: Revenir dessus pour mieux comprendre les embeddings de position.

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    text, batch_size=8, max_length=max_length,
   stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)