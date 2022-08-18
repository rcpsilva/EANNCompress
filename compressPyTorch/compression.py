# -*- coding: utf-8 -*-
"""compression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11OnLUeErxqcu6FcORgOcyM6ptlUl_dbt

# REDE NEURAL CONVOLUCIONAL
"""

# Commented out IPython magic to ensure Python compatibility.
# Implementação e treinamento da rede
import torch
from torch import nn, optim

from torchvision import models
from torchsummary import summary

import copy
import random
import os
#from _typeshed import NoneType

import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.quantization

# Carregamento de Dados
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from torch import optim

# Plots e análises
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time, os

# %matplotlib inline

#from google.colab import drive
#drive.mount('/content/drive')

# Configurando hiperparâmetros.
args = {
    'epoch_num': 20,     # Número de épocas.
    'lr': 1e-3,           # Taxa de aprendizado.
    'weight_decay': 1e-3, # Penalidade L2 (Regularização).
    'batch_size': 50,     # Tamanho do batch.
}

# Definindo dispositivo de hardware
'''
if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

args['device'] = torch.device('cpu')
print(args['device'])
'''

"""## Carregamento de Dados

Usaremos o dataset [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), um conjunto de imagens RGB divididas em 10 categorias de objeto: avião, automóvel, pássaro, gato, veado, cachorro, sapo, cavalo, navio, caminhão. As imagens possuem $32 \times 32$ pixels.

Trata-se de um dataset de 60 mil imagens naturais (do mundo real), muito utilizado para avaliar a qualidade de modelos de aprendizado profundo.

https://pytorch.org/docs/stable/torchvision/datasets.html#cifar
"""

data_transform = transforms.Compose([
                                     transforms.Resize(32),
                                     transforms.ToTensor(),])

train_set = datasets.CIFAR10('.', 
                      train=True, 
                      transform=data_transform, 
                      download=True)

test_set = datasets.CIFAR10('.', 
                      train=False, 
                      transform=data_transform, 
                      download=True)

train_loader = DataLoader(train_set, 
                          batch_size=args['batch_size'], 
                          shuffle=True)

test_loader = DataLoader(test_set, 
                          batch_size=args['batch_size'], 
                          shuffle=True)

# Definindo a rede
net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
#net = net.to(args['device'])

"""#Treino

* **Função de perda**, que vai avaliar a qualidade da performance da rede a cada passo de treinamento;
* **Otimizador**, que a partir da função de perda vai definir a melhor forma de atualizar os pesos.
"""



"""Fluxo de treinamento:

* Iterar nas épocas
* Iterar nos batches
* Cast dos dados no dispositivo de hardware
* Forward na rede e cálculo da loss
* Zerar o gradiente do otimizador
* Cálculo do gradiente e atualização dos pesos

"""

def train(train_loader, net, epoch, criterion, optimizer, device):
 
  # Training mode
  net.train()
  
  start = time.time()
  
  epoch_loss  = []
  pred_list, rotulo_list = [], []
  for batch in train_loader:
    
    dado, rotulo = batch
    
    # Cast do dado na GPU
    dado = dado.to(device)
    rotulo = rotulo.to(device)
    
    # Forward
    ypred = net(dado)
    loss = criterion(ypred, rotulo)
    epoch_loss.append(loss.cpu().data)

    _, pred = torch.max(ypred, axis=1)
    pred_list.append(pred.cpu().numpy())
    rotulo_list.append(rotulo.cpu().numpy())
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
   
  epoch_loss = np.asarray(epoch_loss)
  pred_list  = np.asarray(pred_list).ravel()
  rotulo_list  = np.asarray(rotulo_list).ravel()

  acc = accuracy_score(pred_list, rotulo_list)
  
  end = time.time()
  #print('#################### Train ####################')
  #print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))
  
  return (acc*100)

"""### Validação

Para essa etapa, o PyTorch oferece dois artifícios:
* ```model.eval()```: Impacta no *forward* da rede, informando as camadas caso seu comportamento mude entre fluxos (ex: dropout).
* ```with torch.no_grad()```: Gerenciador de contexto que desabilita o cálculo e armazenamento de gradientes (economia de tempo e memória). Todo o código de validação deve ser executado dentro desse contexto.

Existe o equivalente ao ```model.eval()``` para explicitar que a sua rede deve estar em modo de treino, é o ```model.train()```. Apesar de ser o padrão dos modelos, é boa prática definir também o modo de treinamento.
"""

def validate(test_loader, net, epoch, criterion, device):

  # Evaluation mode
  net.eval()
  
  start = time.time()
  
  epoch_loss  = []
  pred_list, rotulo_list = [], []
  with torch.no_grad(): 
    for batch in test_loader:

      dado, rotulo = batch

      # Cast do dado na GPU
      dado = dado.to(device)
      rotulo = rotulo.to(device)

      # Forward
      ypred = net(dado)
      loss = criterion(ypred, rotulo)
      epoch_loss.append(loss.cpu().data)

      _, pred = torch.max(ypred, axis=1)
      pred_list.append(pred.cpu().numpy())
      rotulo_list.append(rotulo.cpu().numpy())

  epoch_loss = np.asarray(epoch_loss)
  pred_list  = np.asarray(pred_list).ravel()
  rotulo_list  = np.asarray(rotulo_list).ravel()

  acc = accuracy_score(pred_list, rotulo_list)
  
  end = time.time()
  #print('********** Validate **********')
  #print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f\n' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))
  
  return (acc*100, (end-start))

def treinoEteste(Net, device):  
  #train_losses, test_losses = [], []
  #Net = Net.to(args['device'])
  criterion = nn.CrossEntropyLoss().to(device)
  optimizer = optim.Adam(Net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

  for epoch in range(args['epoch_num']):
    
    # Train
    train_losses = train(train_loader, Net, epoch, criterion, optimizer, device)
    
    # Validate
    tempo, test_losses = validate(test_loader, Net, epoch, criterion, device)

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    tamanho = ("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')
    return tamanho

def eval_compression(x, Net, device):
  Net = Net.to(device)
  tamanho =0
  poda = 0
  
  for name, module in Net.named_modules():

    if isinstance(module, torch.nn.Conv2d):
      pesos = np.prod(np.array(module.weight.shape))
      bias = 0 #np.prod(np.array(module.bias.shape))
      tamanho = tamanho + pesos + bias
      
      if int(x[1]) != 0: # vai ter poda para a camada convolucional
        flag = 1
        if int(x[1]) == 1:
          flag = random.randint(0, 2)
        if flag:
          if int(x[5]) == 0: #bias na camada conv
            prune.l1_unstructured(module, name='bias', amount=x[3])
            prune.remove(module, 'bias')
            poda = poda + (bias*x[3])
          elif int(x[5]) == 1: 
            prune.l1_unstructured(module, name='weight', amount=x[3])
            prune.remove(module, 'weight')
            poda = poda + (pesos*x[3])
          else:
            prune.l1_unstructured(module, name='weight', amount=x[3])
            prune.remove(module, 'weight')

            prune.l1_unstructured(module, name='bias', amount=x[3])
            prune.remove(module, 'bias')
            poda = poda + (bias*x[3]) + (pesos*x[3])
            poda = poda + (pesos*x[3])


    elif isinstance(module, torch.nn.Linear):
      pesos = np.prod(np.array(module.weight.shape))
      bias = np.prod(np.array(module.bias.shape))
      tamanho = tamanho + pesos + bias
      if int(x[0]) != 0: # vai ter poda para a camada linear
        flag = 1
        if int(x[0]) == 1:
          flag = random.randint(0, 2)
        if flag:
          if int(x[4]) == 0: #bias na camada linear
            prune.l1_unstructured(module, name='bias', amount=x[2])
            prune.remove(module, 'bias')
            poda = poda + (bias*x[2]) 
          elif int(x[4]) == 1:
            prune.l1_unstructured(module, name='weight', amount=x[2])
            prune.remove(module, 'weight')
            poda = poda + (pesos*x[2])
          else:
            prune.l1_unstructured(module, name='weight', amount=x[2])
            prune.remove(module, 'weight')

            prune.l1_unstructured(module, name='bias', amount=x[2])
            prune.remove(module, 'bias')

            poda = poda + (bias*x[2]) + (pesos*x[2])
           
  redeComPoda = (tamanho - poda)/tamanho
  




  #treinoEteste(Net, device) #realiza o treino da rede

  device = torch.device('cpu')
  criterion = nn.CrossEntropyLoss().to(device)
  Net.to(device)

  #parte da quantização
  if int(x[6]) == 1: #Quantização dinamica
    model_int8 = torch.quantization.quantize_dynamic(
    Net,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
    test_losses, tempo = validate(test_loader, model_int8, 1, criterion, device)
    tamanhoMB= print_model_size(model_int8)

  else: # sem quantização:
    tamanhoMB = print_model_size(Net)
    test_losses, tempo = validate(test_loader, Net, 1, criterion, device)
    
  '''
  elif x[6] == 2: # quantização estatica
    backend = "qnnpack"
    
    Net.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(Net, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    tamanhoMB = print_model_size(model_static_quantized)
    print(tamanhoMB)
    tempo, test_losses = validate(test_loader, model_static_quantized, 1, criterion, device)
  '''
    
 

  return test_losses, redeComPoda #tamanhoMB, , tempo

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


'''
xt = np.zeros((90,7))
cont = 0
x1 = [0, 1, 2]
x2 = [0, 1 ,2]
x3 = [0.1, 0.25, 0.5, 0.75, 0.95]
x7 = [0, 1] #quantização
x = [0, 0, 0, 0, 2, 2, 0]
for i in x1:
  for j in x2:
    for w in x3:
      for y in x7:
        xt[cont][:] = np.array([i, j, w, w, 2, 2, y])
        #print(np.array([i, j, w, w, 2, 2, y]))
        cont = cont + 1 
'''
def comprime(x, op):
  if op == 'resnet50':
    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('compressPyTorch/modelo2.pt', map_location=torch.device('cpu')))
    net3 = copy.deepcopy(net2)
    return eval_compression(x, net3, device)
  elif op == 'vgg16':
    net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    net2 = copy.deepcopy(net)
    net2.load_state_dict(torch.load('compressPyTorch/modelo.pt', map_location=torch.device('cpu')))
    net3 = copy.deepcopy(net2)
    return eval_compression(x, net3, device)
      
'''
for i in range(60, 75):
  net3 = copy.deepcopy(net2)
  #net2 = net2.to(device)
  print("solução: ", i, " ", xt[i])
  print(eval_compression(xt[i], net3, device))

for name, module in net3.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
        print(list(module.named_parameters()))
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):
      print(list(module.named_parameters()))
'''

'''
x1 -> poda para camada linear                                                                                    
     0 -> Não poda                                                        
     1 -> escolhe as camadas de modo aleatório                                      
     2 -> poda todas as camadas

x2 -> poda para camada convolucional
     0 -> Não poda
     1 -> escolhe as camadas de modo aleatório
     2 -> poda todas as camadas

x3 -> Porcentagem de poda para a camada linear

x4 -> Porcentagem de poda para a camada convolucional

x5 -> tipo de poda linear
    0 -> bias
    1 -> weights
    2 -> poda os dois
x6 -> tipo de poda convolucional
    0 -> bias
    1 -> weights
    2 -> poda os dois
x7 -> quantizaçã
    0 -> Sem quantização
    1 -> quantização dinamica
    2 Quantização estatica 

'''

