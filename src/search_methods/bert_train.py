import os
import re
import time
import numpy
import torch
import datetime
from tqdm import tqdm
from transformers import BertTokenizer
from services.preprocess import read_animes_json
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sentence_transformers import SentenceTransformer, models

# set the device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
# torch.cuda.reset_max_memory_allocated()

# top params to tune for bert model
# bert_lr = 2e-5
# max_len = 96
# train_set_size = 1000

bert_lr = 2e-5
max_len = 96
train_set_size = 15444

public_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'public'))

loaded_sim_dataset = numpy.load(os.path.join(
    public_path, 'dataset', 'similarities.npy')).tolist()

print(f"Using {device} device")

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased')


class CosineSimilarityLoss(torch.nn.Module):

    def __init__(self,  loss_fct=torch.nn.MSELoss(), cos_score_transformation=torch.nn.Identity()):

        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.cos = torch.nn.CosineSimilarity(dim=1)

    def forward(self, _input, label):

        embedding_1 = _input[0]
        embedding_2 = _input[1]

        output = self.cos_score_transformation(
            self.cos(embedding_1, embedding_2))

        return self.loss_fct(output, label)


class STSBertModel(torch.nn.Module):

    def __init__(self):

        super(STSBertModel, self).__init__()

        word_embedding_model = models.Transformer(
            'bert-base-uncased', max_seq_length=max_len)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension())
        self.sts_model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model])

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # Pass input pairs through BERT
        params_1 = {'input_ids': input_ids_1,
                    'attention_mask': attention_mask_1}
        params_2 = {'input_ids': input_ids_2,
                    'attention_mask': attention_mask_2}

        outputs1 = self.sts_model(params_1)
        outputs2 = self.sts_model(params_2)

        return outputs1, outputs2


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
        return len([x for x in self.ids1])

    def __getitem__(self, index):
        ids_1 = self.ids1[index]
        ids_2 = self.ids2[index]
        masks_1 = self.masks1[index]
        masks_2 = self.masks2[index]
        labels = self.labels[index]

        return ids_1, ids_2, masks_1, masks_2, labels


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_max_len(texts):
    max_len = 0
    for text in texts:
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    return max_len


def preprocess_text(_texts, m_len):
    curr_labels = []
    input_ids1 = []
    input_ids2 = []
    attention_masks1 = []
    attention_masks2 = []

    for curr_set in loaded_sim_dataset:
        idx1, idx2, sim = curr_set
        idx1 = int(idx1)
        idx2 = int(idx2)

        text1 = re.sub(r'[^a-z ]+', '', _texts[idx1].lower())
        text2 = re.sub(r'[^a-z ]+', '', _texts[idx2].lower())

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

        curr_labels.append(sim)

    return input_ids1, input_ids2, attention_masks1, attention_masks2, curr_labels


# preparing the data ---------------------------------------------------------
data = read_animes_json(os.path.join(public_path, 'dataset', 'animes.json'))

texts = data[1][:train_set_size]

# load half of the data
texts1 = texts[:len(texts)//2]
texts2 = texts[len(texts)//2:]

train_input_ids1, train_input_ids2, train_attention_masks1, train_attention_masks2, labels = preprocess_text(
    texts, max_len)

dataset = TextSimilarityDataset(train_input_ids1, train_input_ids2,
                                train_attention_masks1, train_attention_masks2, labels, max_len)

train_size = int(len(dataset))
val_size = len(dataset) - train_size

batch_size = 32

train_dataloader = DataLoader(
    dataset,
    sampler=RandomSampler(dataset),
    batch_size=batch_size,
)


print("training with", len(loaded_sim_dataset), "x 2 samples")
print(
    f"Max length: {max_len} | # labels: {len(labels)} -----------------")


# exit()
# fine-tune the pre-trained BERT model ----------------------------------------
t_initial = time.time()
epochs = 4

bert_model = STSBertModel()
bert_model.cuda()

criterion = CosineSimilarityLoss()
optimizer = torch.optim.Adam(bert_model.parameters(), lr=bert_lr)

progress_bar = tqdm(range(0, epochs), desc="Training", position=0)

criterion.cuda()

for epoch in progress_bar:

    epoch_time = time.time()
    total_loss_train = 0

    for step, batch in enumerate(train_dataloader):

        b_input_ids1, b_input_ids2, b_input_masks1, b_input_masks2, b_labels = batch

        for i, _ in enumerate(b_input_ids1):

            output1, output2 = bert_model(b_input_ids1[i].to(device), b_input_masks1[i].to(device),
                                          b_input_ids2[i].to(device), b_input_masks2[i].to(device))

            output = [output1['sentence_embedding'],
                      output2['sentence_embedding']]

            curr_label = torch.Tensor([b_labels[i]]).to(
                device=device, dtype=torch.float32)

            loss = criterion(output, curr_label)
            total_loss_train += loss.item()

            progress_bar.set_postfix(
                {"info": f"Epoch {epoch+1}/{epochs} ~ T loss: {total_loss_train/len(dataset):.4f} - {loss.item():.4f} | step {step} of {len(train_dataloader)}"})
            progress_bar.update()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print(
        f"Epoch {epoch} of {epochs} took {format_time(time.time() - epoch_time)}s")
    bert_model.train()


# Measure how long this epoch took.
training_time = format_time(time.time() - t_initial)

print(f"Total training time: {training_time}")

# save the trained bert_model
bert_public_path = os.path.abspath(os.path.join(
    public_path, 'models'))

bert_model.sts_model.save(os.path.join(
    bert_public_path, 'bert_pretrained_model'))

tokenizer.save_pretrained(os.path.join(
    bert_public_path, 'bert_pretrained_tokenizer'))

torch.save(bert_model.state_dict(), os.path.join(
    bert_public_path, 'bert_sts_model.pth'))
