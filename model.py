import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Modelo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        Inicialmos la red neuronal
        
        Parameters:
        input_size : tamaño de la entrada 
        hidden_size : tamaño de la capa escondida
        output_size : tamaño de la salida
        '''
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        '''
        Aplica la funcion de activacion en las capas

        Parameters:
        x : parametros que recibe la red
        Returns:
        x : valores de prediccion
        '''
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def guardar(self, nombre_archivo='model.pth'):
        '''
        Gruarda los datos  del modelo
        '''
        folder_Modelo = './model'
        if not os.path.exists(folder_Modelo): 
            os.makedirs(folder_Modelo) # creamos le folder

        nombre_archivo = os.path.join(folder_Modelo, nombre_archivo)
        torch.save(self.state_dict(), nombre_archivo)


class Entrenar:
    def __init__(self, modelo, lr, gamma):
        '''
        Iniciamos el modeloo con los parametros que creamos correctos
        '''
        self.lr = lr
        self.gamma = gamma
        self.modelo = modelo
        self.optimizer = optim.Adam(modelo.parameters(), lr=self.lr) # Parecido al decenso por gradiente pero con un impulso extra
        self.criterion = nn.MSELoss() #erro medio cuadratico

    def train_step(self, estado, accion, recompensa, sig_estado, done):
        '''
        Se entrena el modelo
        
        Parameters:
        (datos del juego): parametros que recibe la red
        '''

        #pasamos todo a un tensor de pytorch
        estado = torch.tensor(estado, dtype=torch.float)
        sig_estado = torch.tensor(sig_estado, dtype=torch.float)
        accion = torch.tensor(accion, dtype=torch.long)
        recompensa = torch.tensor(recompensa, dtype=torch.float)

        if len(estado.shape) == 1:
            estado = torch.unsqueeze(estado, 0)
            sig_estado = torch.unsqueeze(sig_estado, 0)
            accion = torch.unsqueeze(accion, 0)
            recompensa = torch.unsqueeze(recompensa, 0)
            done = (done, )

        # 1: predicted Q values with current estado
        pred = self.modelo(estado)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = recompensa[idx]
            if not done[idx]:
                Q_new = recompensa[idx] + self.gamma * torch.max(self.modelo(sig_estado[idx]))

            target[idx][torch.argmax(accion[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(accion)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



