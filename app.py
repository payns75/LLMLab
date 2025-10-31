# On va lire le fichier de référence et le stocker dans une variable. C'est le fichier de référence qui sera utilisé pour l'entraînement.
file_path="the-verdict.txt"

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

# Tokenizer du texte, on découpe le texte et les ponctuations.
import re
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
    
from SimpleTokenizerV2Class import SimpleTokenizerV2

tokenizer = SimpleTokenizerV2(vocab)
    
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

ids = tokenizer.encode(text)
decode = tokenizer.decode(ids)
print(ids)
print(decode)