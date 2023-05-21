import copy
import os
import torch
import random
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset

def read_data(file_path, num=None):
    with open(file_path,'r',encoding='utf-8') as f:
        all_data = f.read().split("\n\n")
    if num is not None:
        return all_data[:-1][:num]
    else:
        return all_data[:-1]
class MyDataset(Dataset):
    def __init__(self,all_data, word_2_index):
        self.all_data = all_data
        self.word_2_index = word_2_index

    def __getitem__(self, index):
        text_data = self.all_data[index][:seq_max_len]
        text_data = text_data.split('\n')
        text_idx = []
        for line in text_data:
            text_idx += [word_2_index.get(i, 1) for i in line]
            text_idx += [2]
        input_idx = text_idx[:-1]
        label_idx = text_idx[1:]

        return input_idx, label_idx, len(input_idx)

    def __len__(self):
        return len(all_data)

    def pro_data(self, batch_data):
        batch_input, batch_label, batch_len = zip(*batch_data)
        batch_max_len = max(batch_len)
        new_batch_input = []
        new_batch_label = []

        for input_idx,label_idx in zip(batch_input,batch_label):
            input_idx = input_idx + [0]*(batch_max_len - len(input_idx))
            label_idx = label_idx + [0]*(batch_max_len - len(label_idx))
            new_batch_input.append(input_idx)
            new_batch_label.append(label_idx)
        return torch.tensor(new_batch_input), torch.tensor(new_batch_label)
def get_word_2_index(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        index_2_word = f.read().split('\n')
    word_2_index = {word:idx for idx,word in enumerate(index_2_word)}
    return word_2_index, index_2_word
class Feed_Forward(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(768,1024)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(1024,768)
        self.norm = nn.LayerNorm(768)

    def forward(self, x):
        x1 = self.Linear1(x)
        x1 = self.relu(x1)
        x1 = self.Linear2(x1)

        x = x1 + x
        x = self.norm(x)
        return x
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_num = 4
        self.Que = nn.Linear(768, 768)
        self.Key = nn.Linear(768, 768)
        self.Val = nn.Linear(768, 768)
        self.norm = nn.LayerNorm(768)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,x, attention_mask):
        dim0, dim1, _ = x.shape
        Q = self.Que(x).reshape(dim0, dim1, self.head_num, -1).transpose(1,2)
        K = self.Key(x).reshape(dim0, dim1, self.head_num, -1).transpose(1,2)
        V = self.Val(x).reshape(dim0, dim1, self.head_num, -1).transpose(1,2)
        weight = Q @ K.transpose(-1, -2) / 20
        attention_mask = attention_mask.unsqueeze(3)
        attention_mask = attention_mask.expand_as(weight)
        look_mask = torch.triu(torch.ones_like(attention_mask), 1) == 1
        # mask1 = (look_mask + attention_mask) >= 1
        mask = look_mask | attention_mask
        weight.masked_fill_(mask, -1e9)
        x1 = self.softmax(weight) @ V
        x1 = x.transpose(1,2).reshape(dim0,dim1,-1)
        x = x + x1
        x = self.norm(x)
        return x
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_block1 = MultiHeadAttention()
        self.attention_block2 = MultiHeadAttention()
        self.feed_forward = Feed_Forward()
    def forward(self,x, attention_mask):
        x = self.attention_block1(x,attention_mask)
        x = self.attention_block2(x,attention_mask)
        x = self.feed_forward(x)
        return x
class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_max_len,768)
        self.token_emb = nn.Embedding(vocab_len, 768)
    def forward(self, x):
        seq_len = x.shape[1]
        position = torch.arange(0,seq_len,device=x.device)
        position = position.reshape(1,-1)
        position = position.expand_as(x)
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(position)
        emb = token_emb+pos_emb
        return emb

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        n = 3
        self.layers1 = nn.ModuleList([DecoderBlock() for i in range(n)]) # 需要拆开循环
        #self.layers2 = nn.Sequential(*[DecoderBlock for i in range(n)])  # 可以一键forward
    def forward(self,x):
        attention_mask = get_attention_mask(x)
        x = self.embedding(x)

        for layer in self.layers1:
            x = layer.forward(x,attention_mask)
        return x

class GPT_model(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.decoder = Decoder()
        self.cls = nn.Linear(768, vocab_len)
        self.loss_fun = nn.CrossEntropyLoss()
    def forward(self, x, label=None):
        decoder_out = self.decoder(x)
        pre = self.cls(decoder_out)

        if label is not None:
            loss = self.loss_fun(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)

    def answer1(self,input_text):
        input_idx = [word_2_index.get(i,1) if i!='\n' else 2 for i in input_text]
        input_idx = torch.tensor(input_idx,device=device).reshape(1,-1)

        while True:
            pre = int(self.forward(input_idx)[0][-1])
            input_idx = torch.cat([input_idx,torch.tensor([[pre]],dtype=input_idx.dtype,device=input_idx.device)],-1)
            if pre==2:
                break
        return input_idx[0].tolist()
    def answer2(self,input_text,top_k=5):   # 轮盘赌
        input_idx = [word_2_index.get(i, 1) if i != '\n' else 2 for i in input_text]
        input_idx = torch.tensor(input_idx, device=device).reshape(1, -1)

        while True:
            pre = self.forward(input_idx)
            weight, index = torch.sort(pre, dim=-1)
            # weight = nn.Softmax(dim=-1).forward(weight)
            weight = weight[0][-1].tolist()[:-top_k-1:-1]
            sum_ = sum(weight)

            weight = [int(i/sum_*10) for i in weight]
            index = index[0][-1].tolist()[:-top_k-1:-1]
            random_list = [ i for i,times in zip(index,weight) for j in range(times)]
            pre = random.choice(random_list)
            input_idx = torch.cat([input_idx, torch.tensor([[pre]], dtype=input_idx.dtype, device=input_idx.device)], -1)
            if pre == 2:
                break
        return input_idx[0].tolist()
def get_attention_mask(x):
    padding_position = (x == 0)
    padding_position = padding_position.unsqueeze(1)
    return padding_position
if __name__=="__main__":
    batch_size, epoch = 100, 20
    seq_max_len = 316
    device = "cuda:0"

    data_path = os.path.join("dataset", "train.txt")
    word_2_index, index_2_word = get_word_2_index(os.path.join("dataset","vocab.txt"))
    vocab_len = len(word_2_index)
    all_data = read_data(data_path)
    dataset = MyDataset(all_data,word_2_index)
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False,collate_fn=dataset.pro_data)
    model = GPT_model(len(index_2_word)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)


    for e in range(epoch):
        for batch_idx, batch_label in tqdm(dataloader):
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            loss = model.forward(batch_idx, batch_label)
            loss.backward()

            opt.step()
            opt.zero_grad()

        print(f"loss:{loss:.3f}")
    torch.save(model.state_dict(), "model_weight/myGTP.bin")
    input_text = input("请输入：") + '\n'
    out_idx = model.answer2(input_text)
    stop=[0,1,2]
    out_text = [index_2_word[i] for i in out_idx if i not in stop]
    print(out_text)