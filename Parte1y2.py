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
transform = transforms.Compose([transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

transform = transforms.Compose([transforms.ToTensor()])
#--------------------------Datos para entrenar y testear----------------------------------------------------
train_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = True,  transform = transform)
valid_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
#--------------------------Dividimos conjunto de prueba en datos entrenamiento/validacion-----------------
#train_ds, val_ds = random_split(train_set_orig, [50000, 10000])# Dividimos Conjunto entrada

#---------------------Modificamos Datasets para usar la MSE(toma vectores)---------------------------------
class CustomDataser(Dataset):
    def __init__(self,dataset):
        self.dataset=dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,i):
        image, label = self.dataset[i]
        input= image
        output = image #torch.flatten(image)
        return input,output
    
#-------------------------------------#Creamos los dataloaders--------------------------------------------
""""
train_set =CustomDataser(train_ds)
valid_set =CustomDataser(val_ds)
test_set =CustomDataser(valid_set_orig)
"""
train_set =CustomDataser(train_set_orig)
valid_set =CustomDataser(valid_set_orig)

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

########## Para ayudar en la visualizacion de las imagenes ######
def batch(x):
    return x.unsqueeze(0) #(28,28) -> (1,28,28)

def unbatch(x):
    return x.squeeze().detach().cpu().numpy() #(1,28,28) -> (28,28)

"""    
figure = plt.figure()
rows, cols = 3,2

i = 0
for cols in range(1, cols +1):
    j = torch.randint(len(train_set),size=(1,)).item()
    i += 1
    image,_ = train_set[j]
    figure.add_subplot(rows,cols,i)
    if cols==1:
        plt.title('Original')
    plt.axis('off')
    plt.imshow(unbatch(image),cmap='Greys_r')
    i +=1
    figure.add_subplot(rows,cols,i)
    if cols == 1:
        plt.title('Predicha')
    plt.axis('off')
    image_pred = unbatch(model(batch(image)))
    plt.imshow(image_pred,cmap='Greys_r')
plt.show()
"""

i = 0
model.eval()
with torch.no_grad():
    plt.figure(figsize=(25,5))
    for i in range(5):      
      j = torch.randint(len(valid_set),size=(1,)).item()
      image,_ = valid_set[j]
      plt.subplot(2,5,i+1)
      plt.imshow(unbatch(image),cmap='Greys_r')
      plt.title('Original')
      plt.axis('off')
      image_pred = unbatch(model(batch(image)))
      plt.subplot(2,5,i+6)        
      plt.imshow(image_pred,cmap='Greys_r')
      plt.title('Predicha')
      plt.axis('off')
plt.show()

#--------------------------------------PARTE "2"-----------------------------------------------------------

#2.1)Implemente, en una función, un loop de entrenamiento que recorra los batchs (lotes)
#Entrenamiento
def train_loop(dataloader,model,loss_fn,optimizer):
    model.train()
    num_batches =len(dataloader)
    sum_loss=0
    size = len(dataloader.dataset)
    model.to(device)
    for batch, (X,y) in enumerate(dataloader):
    #calculamos la prediccion del modelo y la correspondiente perdida (error)
        X = X.to(device)
        y = y.to(device)    
        pred= model(X)
        loss=loss_fn(pred,y)#---------------------
    #backpropagamos usando el optimizador proveido
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_batch = loss.item()
        sum_loss += loss_batch
        if batch % 100 == 0:
            current = batch*len(X)
            print(f"@trainloop batch={batch:>5d} loss={loss:>7f}  muestras-procesadas:[{current:>5d}/{size:>5d}]")
    avg_loss = sum_loss / num_batches # loss per batch
    return avg_loss

#2.2)Implemente, en una funci´on, un loop de validación (o prueba) que recorra los batchs.

#Validacion
def valid_loop(dataloader,model,loss_fn):
   model.eval()
   num_batches =len(dataloader)
   sum_los=0
   #loss_value=0
   model.to(device)
   with torch.no_grad():
      for batch_size, (X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        #calculamos la prediccion del modelo y la correspondiente perdida (error)
        pred= model(X)
        loss = loss_fn(pred,y)
        loss_batch=loss.item()
        sum_los +=loss_batch
       #loss_value=loss_fn(pred,true)
      # sum_los+=(1/(batch_size+1))*(loss_value.data.item()-sum_los)
        #backpropagamos usando el optimizador proveido
   avg_los= sum_los/num_batches
   print(f'Test Error: Avg loss: {avg_los:>8f} \n')
   return avg_los


#2.3) Inicialize dos DataLoaders llamados train_loader y valid_loader que estén definidos sobre el train_set (conjunto de entranmiento) y el valid_set (conjunto de validación) de Fashion-MNIST, respectivamente, y que usen batchs de 100 ejemplos.

batchs= 1000
train_loader=DataLoader(train_set, batch_size=batchs, shuffle= True)
valid_loader=DataLoader(valid_set, batch_size=batchs, shuffle= True)
#test_loader=DataLoader(test_set, batch_size=batch_size, shuffle= True)

#2.4) Cree una función de pérdida usando el Error Cuadrático Medio (ECM).

loss_fn=nn.MSELoss() 

#2-5) Cree un optimizador con un learning rate igual a 10−3. Pruebe con ADAM 
learning_rate=0.1
#optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,eps=1e-08,weight_decay=0,amsgrad= False)
optimizer = optim.SGD(model.parameters(), lr = learning_rate,momentum=0.9)

#2-6) Cree una instancia del modelo con n = 64 neuronas en la capa intermedia y dropout p = 0,2.

#n=64
#p=0.2
#model = autoencoder(n,p)

#2-7) Especifique en que dispositivo (device) va a trabajar: en una CPU o en una GPU.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)


#2-8/9) Implemente un loop de entrenamiento y validación que trabaje con el train_loader y el valid_loader respectivamente, usando un número arbitrario de épocas. Este loop debe guardar en dos listas los valores de los promedios del ECM sobre el conjunto de entrenamiento y el de validación, respectivamente.
num_epochs=5
list_avg_train_loss_incorrecta = []
list_avg_train_loss=[]
list_avg_valid_loss=[]

for epoch in range(num_epochs):
    print(f'Época {epoch+1}\n............')
    avg_train_loss_incorrecta= train_loop(train_loader,model,loss_fn,optimizer)
    avg_train_loss= valid_loop(train_loader,model,loss_fn) 
    avg_valid_loss=valid_loop(valid_loader,model,loss_fn)       
    list_avg_train_loss_incorrecta.append(avg_train_loss_incorrecta)
    list_avg_train_loss.append(avg_train_loss)
    list_avg_valid_loss.append(avg_valid_loss)    
print("Done!")

with open('train1.pickle', 'wb') as f1:
    pickle.dump(list_avg_train_loss_incorrecta, f1)

with open('trainL1.pickle', 'wb') as f2:
    pickle.dump(list_avg_train_loss, f2)

with open('valid1.pickle', 'wb') as f3:
    pickle.dump(list_avg_valid_loss, f3)


#2-10) Use las listas del inciso anterior para graficar simultaneamente, y en función de las épocas, el ECM de entrenamiento y el ECM

n = len(list_avg_train_loss_incorrecta)
figure = plt.figure()
plt.xlabel('epoch')
plt.ylabel('loss')
#plt.plot(np.arange(n)+0.5,list_avg_train_loss_incorrecta,label='train-inco')
plt.plot(list(range(1,len(list_avg_train_loss)+1)),list_avg_train_loss,label='Loss Train')
plt.plot(list(range(1,len(list_avg_valid_loss)+1)),list_avg_valid_loss,label='Loss Valid',linestyle='--',c='blue')
plt.title('')
plt.legend()
plt.show()

figure = plt.figure()
rows, cols = 3,2
model.eval()
i = 0
for row in range(1, rows +1):
    j = torch.randint(len(valid_set),size=(1,)).item()
    i += 1
    image,_ = valid_set[j]
    figure.add_subplot(rows,cols,i)
    if row==1:
        plt.title('Original')
    plt.axis('off')
    plt.imshow(unbatch(image),cmap='Greys_r')
    i +=1
    figure.add_subplot(rows,cols,i)
    if row == 1:
        plt.title('Predicha')
    plt.axis('off')
    image_pred = unbatch(model(batch(image)))
    plt.imshow(image_pred,cmap='Greys_r')
plt.show()

filename_model=model_name+"-model.pth"
torch.save(model.state_dict(),filename_model)
print("saved Pytorch Model State  "+filename_model)
