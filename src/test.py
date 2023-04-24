import os
import re
import time
import torch
import datetime
import transformers
from tqdm import tqdm
from preprocess import read_animes_json
from transformers import BertModel, BertTokenizer
from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler, random_split


# set the device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

print(f"Using {device} device")

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased')

# Freeze all BERT model parameters
# for param in model.parameters():
#     param.requires_grad = False

# Define dataset class


class TextSimilarityDataset(Dataset):
    def __init__(self, ids1, ids2, masks1, masks2, _labels, _max_len):
        self.ids1 = ids1
        self.ids2 = ids2
        self.masks1 = masks1
        self.masks2 = masks2
        self.labels = _labels
        self.tokenizer = tokenizer
        self.max_len = _max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ids_1 = self.ids1[index]
        ids_2 = self.ids2[index]
        masks_1 = self.masks1[index]
        masks_2 = self.masks2[index]
        labels = self.labels[index]

        return ids_1, ids_2, masks_1, masks_2, labels

# Define siamese network architecture


class SiameseBERT(torch.nn.Module):
    def __init__(self):
        super(SiameseBERT, self).__init__()
        self.bert = model.to(device)
        self.classifier = torch.nn.Linear(
            in_features=1, out_features=1, device=device)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # Pass input pairs through BERT
        outputs1 = self.bert(input_ids=input_ids_1,
                             attention_mask=attention_mask_1)
        outputs2 = self.bert(input_ids=input_ids_2,
                             attention_mask=attention_mask_2)

        # print("Outputs 1 -----------", outputs1)
        # Extract pooled output vectors from BERT
        pooled_output1 = outputs1.pooler_output
        pooled_output2 = outputs2.pooler_output

        # Compute similarity score using dot product
        similarity_function = torch.nn.CosineSimilarity(dim=1)
        # s_score = torch.sum(pooled_output1 * pooled_output2, dim=1)
        s_score = similarity_function(pooled_output1, pooled_output2)

        # print("s_score -----------", s_score)

        # # Pass similarity score through linear layer and sigmoid activation
        s_score = self.classifier(s_score)
        s_score = self.sigmoid(s_score)

        return s_score


bert_model = SiameseBERT().to(device)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_max_len(texts):
    max_len = 0
    for text in texts:
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    return max_len

# preprocess the text data


def preprocess_text(texts_1, texts_2, m_len):
    input_ids1 = []
    input_ids2 = []
    attention_masks1 = []
    attention_masks2 = []

    for i, _ in enumerate(texts_1):
        text1 = re.sub(r'[^a-z ]+', '', texts_1[i].lower())
        text2 = re.sub(r'[^a-z ]+', '', texts_2[i].lower())
        # limit max_len to 128 to prevent long sequence error
        encoded_dict_1 = tokenizer(
            text1,
            add_special_tokens=True,
            max_length=m_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )
        encoded_dict_2 = tokenizer(
            text2,
            add_special_tokens=True,
            max_length=m_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )

        input_ids1.append(encoded_dict_1['input_ids'])
        input_ids2.append(encoded_dict_2['input_ids'])
        attention_masks1.append(encoded_dict_1['attention_mask'])
        attention_masks2.append(encoded_dict_2['attention_mask'])

    return input_ids1, input_ids2, attention_masks1, attention_masks2


# example usage
# preparing the data ---------------------------------------------------------
data = read_animes_json()

texts = data[1][:]

# load half of the data
texts1 = texts[:len(texts)//2]
texts2 = texts[len(texts)//2:]

# text = texts[0].lower()
# text = re.sub(r'[^a-z ]+', '', text)
# print('example text: ', text)
# print('example tokenized text: ', tokenizer.encode_plus(texts[0], add_special_tokens=True,
#                                                         max_length=80,
#                                                         padding='max_length',
#                                                         return_attention_mask=True,
#                                                         truncation=True,
#                                                         return_tensors='pt'))

labels = [1 for i in range(0, len(texts1))]

max_len = 80

train_input_ids1, train_input_ids2, train_attention_masks1, train_attention_masks2 = preprocess_text(
    texts1, texts2, max_len)

# dataset = Dataset(train_input_ids1, train_input_ids2,
#                   train_attention_masks1, train_attention_masks2, labels)

dataset = TextSimilarityDataset(train_input_ids1, train_input_ids2, train_attention_masks1,
                                train_attention_masks2, labels, max_len)

train_size = int(len(dataset))
val_size = len(dataset) - train_size

# train_dataset, _ = random_split(dataset, [train_size, val_size])

batch_size = 32

# train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

train_dataloader = DataLoader(
    dataset,
    sampler=RandomSampler(dataset),
    batch_size=batch_size
)

total_loss = 0
t_initial = time.time()

print("training with", train_size, "samples")
print(
    f"Max length: {max_len} | labels: {len(labels)} -----------------")


# exit()
# fine-tune the pre-trained BERT model ----------------------------------------
epochs = 5

# Set up optimizer and learning rate scheduler
optimizer = torch.optim.Adam(bert_model.parameters(), lr=2e-5)
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

progress_bar = tqdm(range(0, epochs), desc="Training", position=0)

for epoch in progress_bar:

    bert_model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 10 == 0 and not step == 0:
            elapsed = format_time(time.time() - t_initial)
            print(
                f'Batch {step} of {len(train_dataloader)}. time: {elapsed}s')

        # print("passou por aqui -----------------------------------------------------")

        b_input_ids1, b_input_ids2, b_input_masks1, b_input_masks2, b_labels = batch

        # print(f"b_input_ids1: --------- {b_input_ids1}\n " +
        #       f"b_input_ids2: --------- {b_input_ids2}\n " +
        #       f"b_input_masks1: --------- {b_input_masks1}\n " +
        #       f"b_input_masks2: --------- {b_input_masks2}\n " +
        #       f"b_labels: --------- {b_labels}\n")
        batch_loss = 0

        for i, _ in enumerate(b_input_ids1):
            input_ids = [b_input_ids1[i], b_input_ids2[i]]
            # print(f"input_ids: --------- {input_ids}\n ")

            attention_masks = [b_input_masks1[i], b_input_masks2[i]]

            sim_score = bert_model.forward(input_ids[0].to(device), attention_masks[0].to(device),
                                           input_ids[1].to(device), attention_masks[1].to(device))

            label = torch.Tensor([b_labels[i]]).to(
                device=device, dtype=torch.float32)

            # print(f"sim_score: --------- {sim_score}")
            # print(f"label: --------- {label}\n ")

            loss = torch.nn.functional.binary_cross_entropy(sim_score, label)

            loss.backward()

            total_loss += loss.item()
            batch_loss = loss.item()

            torch.nn.utils.clip_grad_norm_(
                bert_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(
                {"loss": f"{batch_loss:.4f}"})

            progress_bar.update()

    print(f"total epoch time: {format_time(time.time() - t_initial)}")
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.4f}")


avg_train_loss = total_loss / train_size

# Measure how long this epoch took.
training_time = format_time(time.time() - t_initial)

print("")
print(f"Average training loss: {avg_train_loss:.2f}")
print(f"Total training time: {training_time}")

# save the trained bert_model
public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public'))

# bert_model.save_pretrained(
#     os.path.join(public_path, 'bert_pretrained_model'))

torch.save(bert_model.state_dict(), os.path.join(
    public_path, 'bert_pretrained_model.pth'))

# # Save the model to a local file
# torch.save(model.state_dict(), 'model.pth')

# # Load the model from the saved file
# model.load_state_dict(torch.load('model.pth'))


# np.save(os.path.join(public_path, 'bert.text_vectors.npy'), vectors)

# with open(os.path.join(public_path, 'bert.text_vectors.json'), 'w', encoding='utf-8') as f:
#     json.dump(vectors.tolist(), f)
