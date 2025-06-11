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

# Obtener máscara booleana donde t_agua_0 >= 1000
mask = t_agua_0 >= 1000

# Filtrar T_agua usando la máscara y calcular la media
T_agua_eq = round(np.mean(T_agua[mask]),1)

print(f"Temperatura media del agua desde t=1000 s: {T_agua_eq:.2f} °C")

fig, ax=plt.subplots(figsize=(8,4),constrained_layout=True)
ax.plot(t_agua_0,T_agua,label='Agua')
ax.axhline(T_agua_eq,0,1,c='tab:red',ls='--',label='T$_{eq}$ = '+f'{T_agua_mean:.1f} °C')
ax.grid()
ax.set_xlim(0,t_agua_0[-1])
ax.legend()
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
plt.show()
#%% Busco indices donde T cruza la Teq en c/caso
t_cruce_FF1 = np.nonzero(T_FF1==T_agua_eq)[0]
t_cruce_FF2 = np.nonzero(T_FF2==T_agua_eq)[0]
t_cruce_FF3 = np.nonzero(T_FF3==T_agua_eq)[0]
t_cruce_FF4 = np.nonzero(T_FF4==T_agua_eq)[0]
# %%
fig, ax=plt.subplots(figsize=(8,4),constrained_layout=True)
ax.plot(t_FF1_0,T_FF1,label='FF1')
ax.scatter(t_FF1_0[t_cruce_FF1],T_FF1[t_cruce_FF1],label='FF1')
# ax.axhline(T_agua_eq,0,1,c='tab:red',ls='--',label='T$_{eq}$ = '+f'{T_agua_mean:.1f} °C')
ax.set_xlim(0,t_FF1_0[-1])

ax.grid()
ax.legend()
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
plt.show()