from flask import Flask, request, render_template

app = Flask(__name__)

h_size = 512
s_len = 100
number_layers = 3
lr = 0.002
op_s_len = 200
save_path = "model.h5"
data_path = "datasetsirius.txt"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Читаем датасет и переводим его в два массива: один для получения ID-кода символа по самому символу, а второй наоборот: по ID-коду получить символ.
data = open(data_path, 'r', encoding='utf-8').read()
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print(f"В датасете {data_size} символов, из них {vocab_size} уникальных!")

char_to_id = { ch:i for i,ch in enumerate(chars) }
id_to_char = { i:ch for i,ch in enumerate(chars) }

data = list(data)
for i, ch in enumerate(data):
  data[i] = char_to_id[ch]

# Превращаем массив букв в массив чисел, понятный нейросети.
data = torch.tensor(data).to(device)
data = torch.unsqueeze(data, dim=1)

class TPS2(nn.Module):
  def __init__(self, i_size, o_size, h_size, number_layers):
    super(TPS2, self).__init__()
    self.embedding = nn.Embedding(i_size, i_size) # Embedding слой, о котором я говорил в статье
    self.rnn = nn.LSTM(input_size=i_size, hidden_size=h_size, num_layers=number_layers) # LSTM слой
    self.decoder = nn.Linear(h_size, o_size)
  def forward(self, input_seq, memory): # Эта функция выдает следующую букву, пропуская данные (input_seq) через все слои сети, при этом вторым параметром мы передаем "память" сети (memory).
    embedding = self.embedding(input_seq)
    res, new_memory = self.rnn(embedding, memory)
    result = self.decoder(res)
    return result, (new_memory[0].detach(), new_memory[1].detach()) # Возвращаем результат и новое значение памяти нашей сети


tps2 = TPS2(vocab_size, vocab_size, h_size, number_layers).to(device)
tps2.load_state_dict(torch.load(save_path, map_location=device))

def generate_text():
    indexed = np.random.randint(data_size - 1)
    input_data = data[indexed: indexed + 1]
    memory = None
    result = ""
    keep = 0
    while keep < 2:
        # Про этот кусок кода можно прочитать вверху, там где происходит генерация.
        output, memory = tps2(input_data, memory)

        output = F.softmax(torch.squeeze(output), dim=0)
        categ = Categorical(output)
        char = categ.sample()

        result += id_to_char[char.item()]
        if result[-1] == '\n':
            keep+=1
        input_data[0][0] = char.item()

    return result.split('\n')[1]

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html', gen_text=generate_text())

app.run(debug=True)