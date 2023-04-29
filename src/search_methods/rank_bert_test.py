
import os
import re
import torch
import numpy
from tqdm import tqdm
from services.preprocess import read_animes_json
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, models
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
# torch.cuda.reset_max_memory_allocated()
max_len = 96
train_set_size = 1000
print(f"Using {device} device")

# save the trained bert_model
bert_public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'models'))

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(
    os.path.join(bert_public_path, 'bert_pretrained_model'), local_files_only=True)

word_embedding_model = models.Transformer(
    os.path.join(bert_public_path, 'bert_pretrained_model'), max_seq_length=max_len)


pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension())

bert_model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model])

state_dict = torch.load(os.path.join(
    bert_public_path, 'bert_sts_model.pth'))

bert_model.load_state_dict(state_dict, strict=False)

bert_model.eval()


public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public', 'dataset'))

data = read_animes_json(os.path.join(public_path, 'animes.json'))

titles = data[0]
documents = data[1][:train_set_size]

encoded_texts = []

for text in documents:
    text = re.sub(r'[^a-z ]+', '', text.lower())

    encoded_text = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt').to(device)

    encoded_texts.append(encoded_text)

# Define the search phrase and list of texts
# search_phrase = "the zodiacs are the main focus of the anime"
# search_phrase = "the volleyball team is the main focus of the anime"
search_phrase = "a hero who can defeat any enemy with one punch"
# search_phrase = "it has a dragon wich give three wishes to the one who find it"
# search_phrase = "two brothers enter army to become alchemist"
# search_phrase = "a ninja boy who wants to become the leader of his village"
# search_phrase = "give me an anime about giant robots"
# search_phrase = "the protagonist got the shinigami sword and now he has to kill hollows"
# search_phrase = "the protagonist dies and reincarnates as a slime"

search_phrase = re.sub(r'[^a-z ]+', '', search_phrase.lower())

# Tokenize the search phrase and texts
encoded_search = tokenizer(
    search_phrase,
    add_special_tokens=True,
    max_length=max_len,
    padding='max_length',
    truncation=True,
    return_tensors='pt').to(device)


print('shape search: ', encoded_search['input_ids'].shape)
print('shape texts: ', len(encoded_texts))

embed_texts = []

coss = torch.nn.CosineSimilarity()

bert_model.to(device)
# Pass the encoded search phrase and texts through the model and calculate the cosine similarity
with torch.no_grad():
    search_input_ids = encoded_search["input_ids"].to(device)
    search_mask = encoded_search["attention_mask"].to(device)
    params = {'input_ids': search_input_ids, 'attention_mask': search_mask}
    s_curr_encoded = bert_model(params)
    embed_search = s_curr_encoded['sentence_embedding'].cpu()

progress_bar = tqdm(enumerate(encoded_texts), desc="calculating", position=0)

cosine_similarities = []

with torch.no_grad():
    for i, encoded_text in progress_bar:
        input_ids = encoded_text["input_ids"].to(device)
        text_mask = encoded_text["attention_mask"].to(device)
        text_params = {'input_ids': input_ids, 'attention_mask': text_mask}
        curr_encoded = bert_model(text_params)
        embed_text = curr_encoded['sentence_embedding'].cpu()

        # print('embed_text: ', embed_text)
        # exit()
        embed_texts.append(embed_text)

        cosine_similarities.append(coss(embed_search, embed_text))

        progress_bar.update()


print('shape embed_search: ', embed_search.shape)
print('shape embed_texts: [', len(embed_texts), '] ', embed_texts[0].shape)

# Get the indices of the top 10 results for both similarity and euclidean distance
top_cosine_similarities_indices = numpy.argsort(cosine_similarities)[
    :: -1][: int(10)]

# # Print the top 10 results for cosine similarity
# print("top text: ", " ".join(
#     documents[top_cosine_similarities_indices[6]].split()[:96]))
print("Top 10 results for cosine similarity:")
for i, index in enumerate(top_cosine_similarities_indices):
    print(
        f"{i+1}. title: {titles[index]}, Cosine similarity: {cosine_similarities[index]}")
