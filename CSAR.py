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
#%% Agua y Teq
# Obtener máscara booleana donde t_agua_0 >= 1000
mask = t_agua_0 >= 1000

# Filtrar T_agua usando la máscara y calcular la media
T_agua_eq = round(np.mean(T_agua[mask]),1)

print(f"Temperatura media del agua desde t=1000 s: {T_agua_eq} °C")

fig, ax=plt.subplots(figsize=(8,4),constrained_layout=True)
ax.plot(t_agua_0,T_agua,label='Agua')
ax.axhline(T_agua_eq,0,1,c='tab:red',ls='--',label='T$_{eq}$ = '+f'{T_agua_eq:.1f} °C')
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
# %%
#%% Función de ajustes alrededor de Teq
def ajustes_alrededor_Teq(Teq, t, T, x=1.0):
    """
    Realiza ajustes lineal y exponencial alrededor de Teq ± x.
    
    Args:
        Teq (float): Temperatura de equilibrio
        t (np.array): Array de tiempos
        T (np.array): Array de temperaturas
        x (float): Rango alrededor de Teq (default=1.0)
    """
    # Crear máscara para el intervalo de interés
    mask = (T >= Teq - x) & (T <= Teq + x)
    t_interval = t[mask]
    T_interval = T[mask]
    
    # Ajuste lineal
    coeff_lin = np.polyfit(t_interval, T_interval, 1)
    poly_lin = np.poly1d(coeff_lin)
    r2_lin = np.corrcoef(T_interval, poly_lin(t_interval))[0,1]**2
    
    # Ajuste exponencial (T = a + b*exp(-c*t))
    try:
        from scipy.optimize import curve_fit
        def exp_func(t, a, b, c):
            return a + b * np.exp(-c * t)
        
        # Estimación inicial para mejor convergencia
        p0 = [Teq, x, 1/(t_interval[-1] - t_interval[0])]
        popt, pcov = curve_fit(exp_func, t_interval, T_interval, p0=p0)
        a_exp, b_exp, c_exp = popt
        r2_exp = np.corrcoef(T_interval, exp_func(t_interval, *popt))[0,1]**2
        exp_success = True
        
    except Exception as e:
        print(f"Error en ajuste exponencial: {e}")
        exp_success = False
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    ax.plot(t, T, '.-',label='Datos originales')
    #ax.plot(t_interval, T_interval, 'o', label=f'Datos en T_eq ± {x}°C')
    
    # Plotear ajustes
    t_fine = np.linspace(t_interval.min()-50, t_interval.max()+50, 100)
    ax.plot(t_fine, poly_lin(t_fine), '-', 
            label=f'Ajuste lineal: {coeff_lin[0]:.3f}t + {coeff_lin[1]:.3f} (R²={r2_lin:.3f})')
    
    if exp_success:
        ax.plot(t_fine, exp_func(t_fine, *popt), ':',
                label=f'Ajuste exp: {a_exp:.3f} + {b_exp:.3f}exp(-{c_exp:.3f}t) (R²={r2_exp:.3f})')
    
    # ax.axhline(Teq, color='r', linestyle='--', label=f'T_eq = {Teq}°C')
    
    ax.axhspan(Teq-x, Teq+x,0,1,color='tab:red',alpha=0.5,label='$\Delta T$= $\pm$1 ºC')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('T (°C)')
    ax.grid()
    ax.legend()
    ax.set_xlim(400,600)
    ax.set_ylim(T_interval[0]-3,T_interval[-1]+3)
    plt.show()
    
    # Imprimir resultados
    print("\nResultados del ajuste lineal:")
    print(f"Pendiente: {coeff_lin[0]:.5f} °C/s")
    print(f"Ordenada: {coeff_lin[1]:.5f} °C")
    print(f"Coeficiente R²: {r2_lin:.5f}")
    
    if exp_success:
        print("\nResultados del ajuste exponencial:")
        print(f"T_inf: {a_exp:.5f} °C")
        print(f"Amplitud: {b_exp:.5f} °C")
        print(f"Tasa decaimiento: {c_exp:.5f} 1/s")
        print(f"Tau: {1/c_exp:.5f} s")
        print(f"Coeficiente R²: {r2_exp:.5f}")

# Aplicar la función a tus datos
ajustes_alrededor_Teq(T_agua_eq, t_FF1_0, T_FF1, x=1.0)
# %%
