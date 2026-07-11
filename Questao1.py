import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

#
# PARAMETROS DADOS
#

# Unidades:
# Rb: ohms
# g, Rt, D: metros
# alpha_r, theta_r: radianos
# J: kg.m^2
# mi: H/m

alpha_r =  math.pi / 3      # Comprimento angular dos polos do rotor e do estator
g = 0.00045                 # Comprimento do Entreferro
l = 0.2                     # Comprimento medio do caminho magnetico no nucleo
Rt = 0.063                  # Raio do rotor
D = 0.08                    # Comprimento axial da Maquina
N = 90                      # Numero de Espiras
theta_r = 0                 # Posicao angular do rotor em relacao ao estator
Rb = 0.2                    # Resistencia eletrica total da bobina e do sistema de alimentacao
J = 0.010                   # Momento de inercia do rotor

# constantes:
mi = 4 * math.pi * 10**-7   # permeabilidade do vacuo


# ==============================
# TABELA B x H
# ==============================

H_tab = np.array([
    0, 68, 135, 203, 271, 338, 406, 474, 542, 609,
    1100, 1500, 2500, 4000, 5000, 9000, 12000, 20000, 25000
], dtype=float)

B_tab = np.array([
    0, 0.733, 1.205, 1.424, 1.517, 1.560, 1.588, 1.617,
    1.631, 1.646, 1.689, 1.703, 1.724, 1.731, 1.738,
    1.761, 1.770, 1.800, 1.816
], dtype=float)

# Interpolacao:

B_de_H = PchipInterpolator(H_tab, B_tab)

H_de_B = PchipInterpolator(B_tab, H_tab)



# Fiz Esse grafico para verificar se a interpolacao esta correta
# Nele e possivel ver que o modelo nao e linear e satura a partir 
# de certo ponto, o que e esperado para materiais ferromagneticos

 
H_plot = np.linspace(0, 25000, 1000)
B_plot = B_de_H(H_plot)


plt.figure()
plt.plot(H_tab, B_tab, 'o', label='Dados da tabela')
plt.plot(H_plot, B_plot, label='Interpolação PCHIP')
plt.xlabel('H [A/m]')
plt.ylabel('B [T]')
plt.title('Curva B x H do material magnético')
plt.grid(True)
plt.legend(["Dados da tabela", "Interpolação PCHIP"])
plt.savefig("grafico_BH.png", dpi=300, bbox_inches="tight")



# =========================================
# Equacoes do modelo matematico da maquina
# =========================================


# Angulo de Sobreposicao em funcao da posicao angular do rotor
def beta(theta):
    """
    Calcula o angulo de sobreposicao entre o rotor e o estator em funcao da posicao angular do rotor.
    :param theta: Posicao angular do rotor em relacao ao estator (rad)
    :return: Angulo de sobreposicao entre o rotor e o estator (rad)
    """
    return max(alpha_r - abs(theta), 0.0)

# Area de Sobreposiao em funcao da posicao angular do rotor
def Area_entreferro(theta):
    """
    Calcula a area de sobreposicao entre o rotor e o estator em funcao da posicao angular do rotor.
    :param theta: Posicao angular do rotor em relacao ao estator (rad)
    :return: Area de sobreposicao entre o rotor e o estator (m^2)
    """
    return D * Rt * beta(theta)
    
A_max = D * Rt * alpha_r
A_c = A_max




def fluxo_magnetico(i, theta):
    """
    Resolve o fluxo magnético Phi para uma dada corrente i
    e posição angular theta.
    """

    Ag = Area_entreferro(theta)

    # Se não há sobreposição, considera fluxo nulo no modelo ideal
    if Ag <= 0:
        return 0.0

    def equacao(phi):
        Bc = phi / A_c
        Bg = phi / Ag

        # Evita extrapolação além da tabela
        if Bc > B_tab[-1]:
            Bc = B_tab[-1]

        Hc = float(H_de_B(Bc))

        queda_nucleo = Hc * l
        queda_entreferro = (2 * g * Bg) / mi

        return queda_nucleo + queda_entreferro - N * i

    # Limite superior baseado no maior B da tabela
    phi_max = A_c * B_tab[-1]
    
    if i == 0:
        return 0.0

    return brentq(equacao, 0.0, phi_max)



def fluxo_concatenado(i, theta):
    phi = fluxo_magnetico(i, theta)
    return N * phi



# ==============================
# TESTE: FLUXO CONCATENADO
# ==============================


angulos_graus = np.arange(-60, 61, 10)
correntes = np.linspace(0, 65, 300)

plt.figure()

for ang in angulos_graus:
    theta = math.radians(ang)

    lambdas = np.array([
        fluxo_concatenado(i, theta)
        for i in correntes
    ])

    plt.plot(correntes, lambdas, label=f'{ang}°')

plt.xlabel('Corrente i [A]')
plt.ylabel('Fluxo concatenado λ [Wb.espira]')
plt.title('Fluxo concatenado em função da corrente')
plt.grid(True)
plt.legend()
plt.savefig("grafico_Fluxo_concatenado.png", dpi=300, bbox_inches="tight")