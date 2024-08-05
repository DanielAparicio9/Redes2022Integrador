#--------------------------Librerias--------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import random
import os
import pickle




#------------------------------------------------------------------------------
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

transform = transforms.Compose([transforms.ToTensor()])

class autoencoder(nn.Module):
    def __init__(self,n,p):
        super(autoencoder,self).__init__()
        self.n = n
        self.p = p
        self.encoder = nn.Sequential(
            #Primera capa Conv
            nn.Conv2d(1,16,kernel_size=3,padding=0),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.MaxPool2d(2,2),
            #Segunda capa Conv
            nn.Conv2d(16,32, kernel_size=3,padding=0),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.MaxPool2d((2,2)),
            #Lineal
            nn.Flatten(),
            nn.Linear(32*5*5,self.n),
            nn.ReLU(),
            nn.Dropout(self.p)
        )

        self.decoder = nn.Sequential(            
            nn.Linear(self.n,32*5*5),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Unflatten(1,(32,5,5)),
            nn.ConvTranspose2d(32,16,kernel_size=5,stride=2),#,output_padding=1),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.ConvTranspose2d(16,1,kernel_size=4,stride=2),
            #nn.Sigmoid()
            nn.ReLU()
            #nn.Dropout(self.p)
        )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

n=512
p=0.2
model=autoencoder(n,p)    

model_name="autoencoder"
#autoencoder_conv = autoencoder(n=n,p=p)

 
filename_model=model_name+"-model.pth"

#--------------------------------------------------PARTE 3------------------------------------------------------------
########## Para ayudar en la visualizacion de las imagenes ######
def batch(x):
    return x.unsqueeze(0) #(28,28) -> (1,28,28)

def unbatch(x):
    return x.squeeze().detach().cpu().numpy() #(1,28,28) -> (28,28)



#3-1)

class clasificador_conv(nn.Module):
    def __init__(self,autoencoder_conv=None,copy_encoder=True,n=n,p=p):
        super().__init__()
        if autoencoder_conv is None:
          print('Creating encoder')
          self.n = n
          self.p = p
          self.lei = 5
          self.ldo = 5
          self.encoder = nn.Sequential(
            #Primera capa Conv
            nn.Conv2d(1,16,kernel_size=3,padding=0),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.MaxPool2d(2,2),
            #Segunda capa Conv
            nn.Conv2d(16,32, kernel_size=3,padding=0),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.MaxPool2d((2,2)),
            #Parte Lineal
            nn.Flatten(),
            nn.Linear(32*5*5,self.n),
            nn.ReLU(),
            nn.Dropout(self.p)
        )
        else:
            self.n = autoencoder_conv.n
            self.p =autoencoder_conv.p
            if copy_encoder:
                print('copyng provided encoder')
                self.encoder = copy.deepcopy(autoencoder_conv.encoder)
            else:
                print('Using provided encoder')
                self.encoder = autoencoder_conv.encoder
            
            self.clasificador = nn.Sequential(
                nn.Linear(self.n,10),
                nn.ReLU(),
                nn.Dropout(self.p)
            )
    def forward(self,x):
        x = self.encoder(x)
        x = self.clasificador(x)
        return x

n=512
p=0.2


model= autoencoder(n,p).to(device)
model.load_state_dict(torch.load(filename_model))

clasificador_conv = clasificador_conv(autoencoder_conv=model)
model = clasificador_conv



#3-2)Reimplemente las funciones con los loop de entrenamiento y validaci´on adaptados para el problema de clasificación (i.e. hay que incorporarles el c´alculo de la precisi´on).
def train_loop1(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_loss = 0.0
    sum_correct = 0.0
    model.to(device)
    for batch,(X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_batch = loss.item()
        sum_loss += loss_batch
        sum_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 1000 == 0:
            current = batch*len(X)
            print(f"@trainloop batch={batch:>5d} loss={loss:>7f}  muestras-procesadas:[{current:>5d}/{size:>5d}]")
    avg_loss = sum_loss/num_batches
    correct = sum_correct/size
    return avg_loss, correct

def valid_loop1(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_loss = 0.0
    sum_correct = 0.0
    model.to(device)
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss_batch = loss_fn(pred,y).item()
            sum_loss += loss_batch
            sum_correct += (pred.argmax(1) == y).type(torch.float).sum().item() 
    avg_loss = sum_loss/num_batches
    #print(f'@validloop avg_loss={avg_loss:8f}')
    frac_correct = sum_correct/size
    print(f"@validloop Accurary {(100*frac_correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return avg_loss,frac_correct



#3-3) Cree una función de pérdida usando la Cross Entropy Loss (CEL).
loss_fn = nn.CrossEntropyLoss()

#3-4)Cree un optimizador con un learning rate igual a 10−3. Pruebe con ADAM.
learning_rate=0.001
#optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
#optimizer=optim.Adam(model.parameters(),lr=learning_rate,eps=1e-03,weight_decay=0,amsgrad= False)
optimizer=optim.Adam(model.clasificador.parameters(),lr=learning_rate,eps=1e-03,weight_decay=0,amsgrad= False)
#3-5) Cree una instancia del modelo con n = 64 neuronas en la capa intermedia y dropout p = 0,2.


#3-6) Especifique en que dispositivo (device) va a trabajar: en una CPU o en una GPU.


#3-7/8)


batchs_size = 100
num_epochs = 100

train_set_ori = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
valid_set_ori = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)

trainloader = DataLoader(train_set_ori, batch_size=batchs_size,shuffle=True)
validloader = DataLoader(valid_set_ori, batch_size=batchs_size,shuffle=True)

list_avg_train_loss = []
list_avg_valid_train_loss =[]
list_avg_valid_loss = []
list_acc_train = []
list_acc_valid_train = []
list_acc_valid = []

for epochs in range(num_epochs):
    print(f'Epoch {epochs +1} \n ...................................................................')
    avg_train_loss, avg_acc_train = train_loop1(trainloader,model,loss_fn,optimizer)
    avg_valid_train_loss, avg_acc_valid_train = valid_loop1(trainloader,model,loss_fn)
    avg_valid_loss, avg_acc_valid = valid_loop1(validloader,model,loss_fn)
    list_avg_train_loss.append(avg_train_loss)
    list_avg_valid_train_loss.append(avg_valid_train_loss)
    list_avg_valid_loss.append(avg_valid_loss)
    list_acc_train.append(avg_acc_train)
    list_acc_valid_train.append(avg_acc_valid_train)
    list_acc_valid.append(avg_acc_valid)
 


with open('trai.pickle', 'wb') as f1:
    pickle.dump(list_avg_valid_train_loss, f1)

with open('valid.pickle', 'wb') as f2:
    pickle.dump(list_avg_valid_loss, f2)

with open('traacc.pickle', 'wb') as f3:
    pickle.dump(list_acc_valid_train, f3)

with open('validacc.pickle', 'wb') as f4:
    pickle.dump(list_acc_valid, f4)

figure = plt.figure()
plt.plot(range(1,num_epochs+1), list_avg_valid_train_loss, label = 'Valid Train Loss')
plt.plot(range(1,num_epochs+1), list_avg_valid_loss, label= 'Valid Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

figure = plt.figure()
plt.plot(range(1,num_epochs+1), list_acc_valid_train, label = 'Valid Train Acc')
plt.plot(range(1,num_epochs+1), list_acc_valid, label= 'Valid Acc')
plt.xlabel('Epochs')
plt.ylabel('Precisión')
plt.legend()
plt.show()

### Para probar el clasificador
subset_indices = torch.randperm(len(valid_set_ori))[:10]
subset = Subset(valid_set_ori,subset_indices)

class_names = ['T-shirt',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle Boot']


validloader_single = DataLoader(subset, batch_size=1, shuffle=True)

model.eval()
correct_predictions = 0
total_samples = len(validloader_single)

plt.figure(figsize=(12, 9))
for i, (inputs, labels) in enumerate(validloader_single):
    inputs, labels = inputs.to(device), labels.to(device)
    
    outputs = model(inputs)
    predicted_label = torch.argmax(outputs, 1).item()

    true_label = labels.item()

    if predicted_label == true_label:
        correct_predictions += 1

    plt.subplot(2, 5, i + 1)
    plt.imshow(unbatch(inputs[0]), cmap='Greys_r')
    plt.title(f'Real: {class_names[true_label]}\nPredicción: {class_names[predicted_label]}')
    plt.axis('off')

accuracy = correct_predictions / total_samples
print(f'Precisión en la clasificación: {accuracy * 100:.2f}%')

plt.show()



y_pred = []
y_true = []

size = len(validloader.dataset)
for batch,(inputs, labels) in enumerate(validloader):
    outputs = model(inputs)
    outputs = (torch.max(torch.exp(outputs),1)[1]).data.cpu().numpy()
    y_pred.extend(outputs)

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)
    if batch %10 == 0:
        current = batch*len(inputs)
        print(f'batch={batch:>5d} muestras procesadas:[{current:>5d}/{size:>5d}')


cf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize = (12,7))
#sns.heatmap(df_cm, annot = True)
sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='crest', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicha')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
