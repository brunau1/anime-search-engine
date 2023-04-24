from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, models

from datasets import load_dataset

dataset = load_dataset("stsb_multi_mt", name="en", split="train")

print(dataset[0])

print(dataset[1])

similarity = [i['similarity_score'] for i in dataset]
normalized_similarity = [i/5.0 for i in similarity]

word_embedding_model = models.Transformer(
    'bert-base-uncased', max_seq_length=96)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension())
sts_bert_model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model])


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence_1 = [i['sentence1'] for i in dataset]
sentence_2 = [i['sentence2'] for i in dataset]
text_cat = [[str(x), str(y)] for x, y in zip(sentence_1, sentence_2)][0]

input_data = tokenizer(text_cat, padding='max_length',
                       max_length=96, truncation=True, return_tensors="pt")

output = sts_bert_model(input_data)

print(output['sentence_embedding'][0].size())

print(output['sentence_embedding'][1].size())


class STSBertModel(torch.nn.Module):

    def __init__(self):

        super(STSBertModel, self).__init__()

        word_embedding_model = models.Transformer(
            'bert-base-uncased', max_seq_length=96)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension())
        self.sts_model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model])

    def forward(self, input_data):

        output = self.sts_model(input_data)

        return output


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, dataset):

        similarity = [i['similarity_score'] for i in dataset]
        self.label = [i/5.0 for i in similarity]
        self.sentence_1 = [i['sentence1'] for i in dataset]
        self.sentence_2 = [i['sentence2'] for i in dataset]
        self.text_cat = [[str(x), str(y)]
                         for x, y in zip(self.sentence_1, self.sentence_2)]

    def __len__(self):

        return len(self.text_cat)

    def get_batch_labels(self, idx):

        return torch.Tensor(self.label[idx])

    def get_batch_texts(self, idx):

        return tokenizer(self.text_cat[idx], padding='max_length', max_length=96, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


def collate_fn(texts):

    num_texts = len(texts['input_ids'])
    features = list()
    for i in range(num_texts):
        features.append(
            {'input_ids': texts['input_ids'][i], 'attention_mask': texts['attention_mask'][i]})

    return features


class CosineSimilarityLoss(torch.nn.Module):

    def __init__(self,  loss_fct=torch.nn.MSELoss(), cos_score_transformation=torch.nn.Identity()):

        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.cos = torch.nn.CosineSimilarity(dim=1)

    def forward(self, input, label):

        embedding_1 = torch.stack([inp[0] for inp in input])
        embedding_2 = torch.stack([inp[1] for inp in input])

        output = self.cos_score_transformation(
            self.cos(embedding_1, embedding_2))

        return self.loss_fct(output, label.squeeze())


def model_train(dataset, epochs, learning_rate, bs):

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    model = STSBertModel()

    criterion = CosineSimilarityLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataset = DataSequence(dataset)
    train_dataloader = DataLoader(
        train_dataset, num_workers=4, batch_size=bs, shuffle=True)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # best_acc = 0.0
    # best_loss = 1000

    for i in range(epochs):

        total_acc_train = 0
        total_loss_train = 0.0

        for train_data, train_label in tqdm(train_dataloader):

            train_data['input_ids'] = train_data['input_ids'].to(device)
            train_data['attention_mask'] = train_data['attention_mask'].to(
                device)
            del train_data['token_type_ids']

            train_data = collate_fn(train_data)

            output = [model(feature)['sentence_embedding']
                      for feature in train_data]

            loss = criterion(output, train_label.to(device))
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(
            f'Epochs: {i + 1} | Loss: {total_loss_train / len(dataset): .3f}')
        model.train()

    return model


EPOCHS = 8
LEARNING_RATE = 1e-6
BATCH_SIZE = 32

# Train the model
trained_model = model_train(dataset, EPOCHS, LEARNING_RATE, BATCH_SIZE)


# Load test data
test_dataset = load_dataset("stsb_multi_mt", name="en", split="test")

# Prepare test data
sentence_1_test = [i['sentence1'] for i in test_dataset]
sentence_2_test = [i['sentence2'] for i in test_dataset]
text_cat_test = [[str(x), str(y)]
                 for x, y in zip(sentence_1_test, sentence_2_test)]

# Function to predict test data


def predict_sts(texts):

    trained_model.to('cpu')
    trained_model.eval()

    test_input = tokenizer(texts, padding='max_length',
                           max_length=96, truncation=True, return_tensors="pt")
    test_input['input_ids'] = test_input['input_ids']
    test_input['attention_mask'] = test_input['attention_mask']
    del test_input['token_type_ids']

    test_output = trained_model(test_input)['sentence_embedding']
    sim = torch.nn.functional.cosine_similarity(
        test_output[0], test_output[1], dim=0).item()

    return sim


print(text_cat_test[420])

print(predict_sts(text_cat_test[420]))