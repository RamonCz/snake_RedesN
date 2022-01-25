import pygame #libreria para  crear el juego 
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direcciones(Enum):
    '''
    Enum con las direcciones que puede dar el juego
    '''
    DERECHA = 1
    IZQUIERDA = 2
    ARRIBA = 3
    ABAJO = 4

Punto = namedtuple('Punto', 'x, y')

# rgb colores
BLANCO = (255, 255, 255)
ROJO = (200,0,0)
NEGRO = (0,0,0)
VERDE = (0,128,0)
VERDE2 = (0,100,0)

BLOCK_SIZE = 20
VELOCIDAD = 40

class SnakeGameAI:

    def __init__(self, ancho=640, alto=480):
        '''
        Inicializacion del objeto juego Snakegame 
        
        Parameters:
        ancho : ancho de la pantalla 
        alto : alto de la pantalla
        '''
        self.ancho = ancho
        self.alto = alto
        # Inicia la pantalla con los valores ancho y alto
        self.pantalla = pygame.display.set_mode((self.ancho, self.alto))
        pygame.display.set_caption('Snake')
        self.reloj = pygame.time.Clock()
        self.reinicio()


    def reinicio(self):
        '''
        Inicio del juego se reinicia el juego desde el principio
        '''
        self.direccion = Direcciones.DERECHA

        self.cabeza = Punto(self.ancho/2, self.alto/2)
        self.snake = [self.cabeza,
                      Punto(self.cabeza.x-BLOCK_SIZE, self.cabeza.y),
                      Punto(self.cabeza.x-(2*BLOCK_SIZE), self.cabeza.y)]

        self.puntaje = 0
        self.comida = None
        self._lugar_comida()
        self.frame_iteration = 0 # numeros de FPS en la iteracion

        
    def _lugar_comida(self):
        '''
        Coloca la comida en algun random del juego
        '''
        x = random.randint(0, (self.ancho-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.alto-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.comida = Punto(x, y)
        if self.comida in self.snake:
            self._lugar_comida()


    def dar_paso(self, accion):
        '''
        Da un paso en el juego haciendo que se mueva snake(vibora)

        Parameters:       
        accion : accion de va a hacer snake 
        '''
        self.frame_iteration += 1
        # 1. obtiene las acciones del usuario
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. mueve
        self._mover(accion) # actualiza la cabeza
        self.snake.insert(0, self.cabeza)
        
        # 3. verifica que el juego no haya termiando
        recompensa = 0
        fin_juego = False
        if self.es_choque() or self.frame_iteration > 100*len(self.snake):
            fin_juego = True
            recompensa = -10
            return recompensa, fin_juego, self.puntaje

        # 4. luegar de la nueva comida o solo se mueve
        if self.cabeza == self.comida:
            self.puntaje += 1
            recompensa = 10
            self._lugar_comida()
        else:
            self.snake.pop()
        
        # 5. actualiza la pantalla y el reloj
        self._actualiza_pantalla()
        self.reloj.tick(VELOCIDAD)
        # 6. da fin el juego y segresa la recompensa
        return recompensa, fin_juego, self.puntaje


    def es_choque(self, pt=None):
        '''
        Informa  si choco con si mismo o alguna pared 
        '''
        if pt is None:
            pt = self.cabeza
        # si le pega al borde 
        if pt.x > self.ancho - BLOCK_SIZE or pt.x < 0 or pt.y > self.alto - BLOCK_SIZE or pt.y < 0:
            return True
        # Si se pega con el mismo
        if pt in self.snake[1:]:
            return True

        return False


    def _actualiza_pantalla(self):
        '''
        Actualiza la pantalla del juego
        '''
        
        self.pantalla.fill(NEGRO) #fondo

        for pt in self.snake:
            pygame.draw.rect(self.pantalla, VERDE2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.pantalla, VERDE, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.pantalla, ROJO, pygame.Rect(self.comida.x, self.comida.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("puntaje: " + str(self.puntaje), True, BLANCO)
        self.pantalla.blit(text, [0, 0])
        pygame.display.flip()


    def _mover(self, accion):
        '''
        Se cambia los valores del snake dependiendo la accion recibida

        Parameters:
        accion: accion que debe tomar
        '''
        # [ADELANTE, DERECHA, IZQUIERDA]

        reloj_apunta = [Direcciones.DERECHA, Direcciones.ABAJO, Direcciones.IZQUIERDA, Direcciones.ARRIBA]
        idx = reloj_apunta.index(self.direccion)

        if np.array_equal(accion, [1, 0, 0]):
            nueva_direc = reloj_apunta[idx] # no cambia
        elif np.array_equal(accion, [0, 1, 0]):
            sig_indx = (idx + 1) % 4
            nueva_direc = reloj_apunta[sig_indx] # DERECHA da vuelta 
        else: # [0, 0, 1]
            sig_indx = (idx - 1) % 4
            nueva_direc = reloj_apunta[sig_indx] # IZQUIERDA da vuelta  

        self.direccion = nueva_direc

        x = self.cabeza.x
        y = self.cabeza.y
        if self.direccion == Direcciones.DERECHA:
            x += BLOCK_SIZE
        elif self.direccion == Direcciones.IZQUIERDA:
            x -= BLOCK_SIZE
        elif self.direccion == Direcciones.ABAJO:
            y += BLOCK_SIZE
        elif self.direccion == Direcciones.ARRIBA:
            y -= BLOCK_SIZE

        self.cabeza = Punto(x, y)