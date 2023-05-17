from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
import torch
from sklearn.model_selection import train_test_split
import os
import json
import re
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from preprocess import bert_simple_preprocess_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

print(f"Using {DEVICE} device")

animes_file_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'animes.json'))

train_data_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset', 'train_dataset_V5.json'))

bert_trained_model_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'models', 'bert_trained_model__v5'))

with open(animes_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(train_data_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)


titles = data['names']  # array com os títulos dos textos
synopses = data['content']  # array com os textos

pairs = []

for i, data in enumerate(train_data):
    # Converte para minúsculas
    text = synopses[data[1]].lower()
    # Remove caracteres especiais e números
    text = re.sub(r'[^a-z ]+', '', text)

    pairs.append([data[0], text, data[2]])

# print("example pair: ", pairs[0], "\n\n")

print("pairs len: ", len(pairs))

# Download the pre-trained retriBERT model and tokenizer ----------------------------


# Define the model. Either from scratch of by loading a pre-trained model
word_embedding_model = models.Transformer(
    "distilroberta-base", max_seq_length=160)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Define your train examples. You need more than just two examples...
# train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#                   InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

train_examples = []
for i, data in enumerate(train_data):
    text = synopses[data[1]]
    text = bert_simple_preprocess_text(text)
    query = bert_simple_preprocess_text(data[0])
    # convert the label to float
    label = float(data[2])

    train_examples.append(
        InputExample(texts=[query, text], label=label))
    # also add the inverse pair
    train_examples.append(
        InputExample(texts=[text, query], label=label))

print("train examples len: ", len(train_examples))

# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=24)
train_loss = losses.MultipleNegativesRankingLoss(model)
# train_loss = losses.CosineSimilarityLoss(model)

model.to(DEVICE)
# Tune the model
model.fit(train_objectives=[
          (train_dataloader, train_loss)], epochs=4, warmup_steps=100, optimizer_params={'lr': 1e-5})

# Save the model
model.save(bert_trained_model_path)
