#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from glob import glob
from datetime import datetime
#%%
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura 
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 

    return timestamp,temp_CH1, temp_CH2
#%%
path_agua='250611_104227_agua.csv'
path_FF1='250611_110618_FF1.csv'
path_FF2='250611_112442_FF2.csv'
path_FF3='250611_114436_FF3.csv'
path_FF4='250611_115754_FF4.csv'

t_agua,T_agua,_=lector_templog(path_agua)
t_FF1,T_FF1,_=lector_templog(path_FF1)
t_FF2,T_FF2,_=lector_templog(path_FF2)
t_FF3,T_FF3,_=lector_templog(path_FF3)
t_FF4,T_FF4,_=lector_templog(path_FF4)

t_agua_0 = np.array([(t-t_agua[0]).total_seconds() for t in t_agua])
t_FF1_0 = np.array([(t-t_FF1[0]).total_seconds() for t in t_FF1])
t_FF2_0 = np.array([(t-t_FF2[0]).total_seconds() for t in t_FF2])
t_FF3_0 = np.array([(t-t_FF3[0]).total_seconds() for t in t_FF3])
t_FF4_0 = np.array([(t-t_FF4[0]).total_seconds() for t in t_FF4])

# %% 
fig, ax=plt.subplots(constrained_layout=True)
ax.plot(t_agua_0,T_agua,label='Agua')
ax.plot(t_FF1_0,T_FF1,label='FF1')
ax.plot(t_FF2_0,T_FF2,label='FF2')
ax.plot(t_FF3_0,T_FF3,label='FF3')
ax.plot(t_FF4_0,T_FF4,label='FF4')
ax.grid()
ax.set_xlim(0,)
#%% voy 1 por 1


# %%
# Calcular las derivadas numéricas dT/dt para cada conjunto de datos
dTdt_agua = np.gradient(T_agua)
dTdt_FF1 = np.gradient(T_FF1)
dTdt_FF2 = np.gradient(T_FF2)
dTdt_FF3 = np.gradient(T_FF3)
dTdt_FF4 = np.gradient(T_FF4)

# Crear figura para los plots de derivadas
plt.figure(figsize=(10, 6))

# Plotear todas las derivadas
plt.plot(t_agua_0, dTdt_agua, label='Agua')
plt.plot(t_FF1_0, dTdt_FF1, label='FF1')
plt.plot(t_FF2_0, dTdt_FF2, label='FF2')
plt.plot(t_FF3_0, dTdt_FF3, label='FF3')
plt.plot(t_FF4_0, dTdt_FF4, label='FF4')

# Añadir elementos gráficos
plt.title('Derivadas dT/dt de las series temporales')
plt.xlabel('Tiempo (s)')
plt.ylabel('dT/dt (°C/s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Mostrar el plot
plt.show()


# %%
