import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direcciones, Punto
from model import Modelo, Entrenar
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate

class Agent:

    def __init__(self):
        self.n_juegos = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memoria = deque(maxlen=MAX_MEMORY) # popleft()
        self.modelo = Modelo(11, 256, 3)
        self.entreno = Entrenar(self.modelo, lr=LR, gamma=self.gamma)


    def obtener_estado(self, game):
        '''
        Toma los datos del juego en dicho momento 
        Parameters:
        game : instancia del juego 

        Returns:
        np.array : valores del estado
        '''
        cabeza = game.snake[0]
        punto_izq = Punto(cabeza.x - 20, cabeza.y)
        punto_der = Punto(cabeza.x + 20, cabeza.y)
        punto_arriba = Punto(cabeza.x, cabeza.y - 20)
        punto_abajo = Punto(cabeza.x, cabeza.y + 20)
        
        dir_izq = game.direccion == Direcciones.IZQUIERDA
        dir_der = game.direccion == Direcciones.DERECHA
        dir_arriba = game.direccion == Direcciones.ARRIBA
        dir_abajo = game.direccion == Direcciones.ABAJO

        estado = [
            # peligro de ir hacia adelante
            (dir_der and game.es_choque(punto_der)) or 
            (dir_izq and game.es_choque(punto_izq)) or 
            (dir_arriba and game.es_choque(punto_arriba)) or 
            (dir_abajo and game.es_choque(punto_abajo)),

            # peligro de ir hacia la derecha
            (dir_arriba and game.es_choque(punto_der)) or 
            (dir_abajo and game.es_choque(punto_izq)) or 
            (dir_izq and game.es_choque(punto_arriba)) or 
            (dir_der and game.es_choque(punto_abajo)),

            # peligro de ir hacia la izquierda
            (dir_abajo and game.es_choque(punto_der)) or 
            (dir_arriba and game.es_choque(punto_izq)) or 
            (dir_der and game.es_choque(punto_arriba)) or 
            (dir_izq and game.es_choque(punto_abajo)),
            
            # movimiento de direccion
            dir_izq,
            dir_der,
            dir_arriba,
            dir_abajo,
            
            # lugar de la comida
            game.comida.x < game.cabeza.x,  # comida izquierda
            game.comida.x > game.cabeza.x,  # comida derecha
            game.comida.y < game.cabeza.y,  # comida arriba
            game.comida.y > game.cabeza.y  # comida abajo
            ]

        return np.array(estado, dtype=int)

    def recuerdo(self, estado, accion, recompensa, sig_estado, done):
        '''
        Guarda todos los datos del juego en la cola para un futuro entrenamiento

        Parameters:
        estado : estado del juego arrays de los datos
        accion : accion que se realizo
        recompensa: recompensa que se lleva hasta ahora
        sig_estado: estado del juego siguiente
        '''
        self.memoria.append((estado, accion, recompensa, sig_estado, done)) # popleft si se excede la memoria

    def entrenamiento_largo(self):
        '''
        Entrena nuestro modelo con todas los recuerdos guardades en la cola       
        '''
        if len(self.memoria) > BATCH_SIZE: # si excedimos la memoria tomamos una parte random
            mini_muestra = random.sample(self.memoria, BATCH_SIZE) # lista de tuplas
        else:
            mini_muestra = self.memoria # tomamos toda 

        estados, acciones, recompensas, sig_estados, dones = zip(*mini_muestra) # descomprime las tuplas de la cola 
        self.entreno.train_step(estados, acciones, recompensas, sig_estados, dones)

    def entrenamiento_corto(self, estado, accion, recompensa, sig_estado, done):
        '''
        Entrena nuestro modelo con una sola accion
        '''
        self.entreno.train_step(estado, accion, recompensa, sig_estado, done)

    def tomar_accion(self, estado):
        # random movimiento: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_juegos
        mov_final = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            mover = random.randint(0, 2)
            mov_final[mover] = 1
        else:
            estado0 = torch.tensor(estado, dtype=torch.float)
            prediction = self.modelo(estado0)
            mover = torch.argmax(prediction).item()
            mov_final[mover] = 1

        return mov_final


def train():
    '''
    Entrena el modelo
    '''
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agente = Agent()
    game = SnakeGameAI()
    while True:
        # toma el ultimo estado
        estado_final = agente.obtener_estado(game)

        # obtiene el ultimo movimiento
        mov_final = agente.tomar_accion(estado_final)

        # mejora el movimiento y obtiene un nuevo estado
        recompensa, done, puntaje = game.dar_paso(mov_final)
        estado_new = agente.obtener_estado(game)

        # entrenamiento con corta memoria 
        agente.entrenamiento_corto(estado_final, mov_final, recompensa, estado_new, done)

        # recuerdo
        agente.recuerdo(estado_final, mov_final, recompensa, estado_new, done)

        if done:
            # train long memory, plot result
            game.reinicio()
            agente.n_juegos += 1
            agente.entrenamiento_largo()

            if puntaje > record:
                record = puntaje
                agente.modelo.guardar()

            print('Juego', agente.n_juegos, 'puntaje', puntaje, 'Record:', record)

            plot_scores.append(puntaje)
            total_score += puntaje
            mean_score = total_score / agente.n_juegos
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()