# -*- coding: utf-8 -*-
"""

Trabalho Final
Tópicos especiais: Python in Environmental Applications
@author: Luiz Felipe Pereira de Brito

"""

# Pacotes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import os
import math
import seaborn as sns
from scipy import stats
#from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import gumbel_r
from matplotlib import cm

# Caminho do diretório
path = r"C:\ENS410064\dados\trabalhoFinal"

# Abrir arquivo dos níveis diários em Manaus
stage_bruto = pd.read_csv(path+'/stage_14990000_MANAUS.csv', sep=';', header=0, parse_dates=['Data'], date_format='%y-%m-%d')
stage_bruto
list_cotas_bruto = stage_bruto['Cota'].tolist()

data = pd.date_range(start='1902-09-01', end='2014-12-31')

fig, ax = plt.subplots()
fig.suptitle('Cotagrama da estação fluviométrica de Manaus')
ax.plot(data, list_cotas_bruto, color='black', linewidth = 0.30)
ax.set_ylabel('Cota (cm)')
ax.set_xlabel('Ano')
plt.show()

# Verificar se tem falha (nesse caso, NaN)
tem_NaN = stage_bruto['Cota'].isna().any().any()
print(f'Tem NaNs no DataFrame? {tem_NaN}')

# Exclui todas as linhas que contêm pelo menos um NaN
stage = stage_bruto.dropna().reset_index()
stage = stage.drop('index', axis=1) # Exclui a coluna 'index'

# Converte a coluna 'Data' para o tipo datetime
stage['Data'] = pd.to_datetime(stage['Data'])

list_cotas = stage['Cota'].tolist()

# Cria o gráfico de cotas para o ano de 2014
stage2014 = stage[stage['Data'].dt.year == 2014]
plt.plot(stage2014['Data'], stage2014['Cota'])
plt.xlabel('Data')
plt.ylabel('Cota (cm)')
plt.title('Cotagrama da estação fluviométrica de Manaus - Ano 2014')
plt.show()

#%% Histograma dos níveis diários em Manaus
N = len(stage)
k = 1 + 3.322*math.log10(N)  # fórmula de Sturges é uma técnica estatística utilizada para determinar o número ideal de classes em um histograma
print(k) #arrendondar para número inteiro mais próximo

plt.hist(list_cotas, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Cotas (cm)')
plt.ylabel('Frequência')
plt.title('Histograma de Cotas Diárias')

# Pega a cota mínima
min(list(filter(lambda x: not math.isnan(x), stage['Cota'])))

# Pega a cota máxima
max(list(filter(lambda x: not math.isnan(x), stage['Cota'])))

#%% Valores médios mensais a partir da série histórica de cotas diárias

# Divide a coluna 'Data' do df 'stage' em três colunas (Ano, Mês, Dia)
stage[['Ano', 'Mês', 'Dia']] = stage['Data'].apply(lambda x: pd.Series([x.year, x.month, x.day]))

# Cria dataframe com a média das cotas de cada mês em determinado ano
df_cotas_mensais = stage.groupby(['Ano', 'Mês'])['Cota'].mean().reset_index()
list_cotas_mensais = df_cotas_mensais['Cota'].tolist()
cota_media = df_cotas_mensais['Cota'].mean()

data_mes = pd.date_range(start='1902-09-01', end='2014-12-31', freq='1M')

fig, ax = plt.subplots()

ax.plot(data_mes, list_cotas_mensais, color='black', label = 'Cota diária em cm', linewidth = 1)
ax.axhline(y=cota_media, color='red', label = 'Cota média', linewidth = 1)

ax.set_ylim([1250, 3250])

for i in range(1250, 3250, 250):
    ax.hlines(y=i, xmin=data_mes[0], xmax=data_mes[-1], colors='gray', linestyles='dashed', alpha=0.5) 
       
artists, labels = ax.get_legend_handles_labels()

ax.legend(artists, labels, loc='upper right', bbox_to_anchor=(0.99, 1.01))

fig.set_size_inches(10,5) # definir o tamanho da figura em polegadas (largura e a altura da figura, respectivamente)

ax.set_xlim(data[0], data[-1])
ax.set_ylabel('Cota (cm)')
ax.set_xlabel('Ano')

plt.show()

# Conta quantos valores na lista "list_cotas_mensais" são menores e maiores ou iguais que "cota_media"
cota_abaixo = sum(1 for cota in list_cotas_mensais if cota < cota_media)
cota_acima = sum(1 for cota in list_cotas_mensais if cota >= cota_media)

#%% Cálculo dos valores médios climatológicos

df_cotas_clima = stage.groupby(['Mês'])['Cota'].mean().reset_index()
list_cotas_clima = df_cotas_clima['Cota'].tolist()

meses = ['JAN','FEV','MAR','MAI','ABR','JUN','JUL','AGO','SET','OUT','NOV','DEZ']

fig, ax = plt.subplots()

ax.bar(meses, list_cotas_clima, color='darkgray', width=0.85, edgecolor='black', label = 'Cota climatológica em Manaus', linewidth = 1)

ax.set_ylim([0, 3000])
     
artists, labels = ax.get_legend_handles_labels()

ax.legend(artists, labels, loc='upper right', bbox_to_anchor=(1, 1))

fig.set_size_inches(10,5)

ax.set_xlim([-0.575,11.575]) # definir os limites do eixo x
ax.set_ylabel('Cota (cm)')
ax.set_xlabel('Mês')

plt.show()

# Mapa de Calor (heatmap)
plt.figure(figsize=(10, 7)) # sazonalidade das cotas
heatmap_data = df_cotas_mensais.pivot_table(index='Ano', columns='Mês', values='Cota', aggfunc='mean')
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False, fmt=".1f", linewidths=.5, cbar_kws={'label': 'Cotas Médias'}) # Use "annot" para representar os valores das células com texto
plt.title('Média Climatológica de Cotas por Ano e Mês')
plt.show()

#%% Boxplot das cotas

# Lista com os meses do ano
meses = ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ']

linhas_log_pri = [1000, 2000, 3000]

linhas_log_sec = [1200, 1400, 1600, 1800,
                  2200, 2400, 2600, 2800,
                  3200, 3400, 3600, 3800]

dados_boxplot = {}
for mes, grupo in stage.groupby('Mês'):
    dados_boxplot[mes] = grupo.dropna()['Cota'].tolist()

plt.figure(figsize=(15, 5))
plt.boxplot(dados_boxplot.values(), boxprops=dict(linewidth=1, color='black'), medianprops=dict(linewidth=1.5, color='red'))
plt.xticks(range(1,len(meses)+1), meses)
plt.ylim([1000,3500])
plt.yscale('log')
for i in linhas_log_sec:
    plt.axhline(y=i, color = 'gray', linestyle='dashed', linewidth=0.5)
for j in linhas_log_pri:
    plt.axhline(y=j, color = 'darkgray', linestyle='solid', linewidth=1)
plt.xlabel('Mês')
plt.ylabel('Cota')

plt.show()

#%% Estabelecer a série de cotas máximas anuais

df_cotas_maximas = stage.groupby('Ano')['Cota'].max().reset_index()

list_cotas_maximas = df_cotas_maximas['Cota'].tolist()

fig, ax = plt.subplots()

# Ajuste de uma reta (y = ax + b) aos dados raster (peguei da internet, não sei como funciona)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_cotas_maximas['Ano'], df_cotas_maximas['Cota'])
line = slope * df_cotas_maximas['Ano'] + intercept
 
# Cálculo do coeficiente de determinação (R²)
r_squared = r_value**2

# Plota a reta de tendência
#plt.plot(df_cotas_maximas['Ano'], line, '-', color='red', label=f'\n \n Reta de Tendência:\n y = {slope:.4f}x + {intercept:.4f}\n R² = {r_squared:.4f}', linewidth=1)
plt.plot(df_cotas_maximas['Ano'], line, color='red', linestyle='-', label='Linha de Tendência', linewidth=1)
ax.plot(df_cotas_maximas['Ano'], df_cotas_maximas['Cota'], color='black', label = 'Cotas máximas em cm', linewidth = 1)

ax.set_xlim([1900, 2015])
ax.set_ylim([2000, 3250])

for i in range(1000, 3300, 200):
    ax.hlines(y=i, xmin=1902, xmax=2014, colors='gray', linestyles='dashed', alpha=0.5) 
       
artists, labels = ax.get_legend_handles_labels()

ax.legend(artists, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95))

fig.set_size_inches(15,5)

ax.set_ylabel('Cota máxima (cm)')
ax.set_xlabel('Ano')

plt.show()

#%% Histograma dos níveis máximos anuais em Manaus

N_max = len(df_cotas_maximas)
k_max = 1 + 3.322*math.log10(N_max)  # fórmula de Sturges é uma técnica estatística utilizada para determinar o número ideal de classes em um histograma
print(k_max) #arrendondar para número inteiro mais próximo

plt.hist(list_cotas_maximas, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Cotas máximas anuais (cm)')
plt.ylabel('Frequência')
plt.title('Histograma de Cotas Máximas Anuais')

#%% Estimativa da distribuição de frequência empírica - Divisão de estações do ano
# Kernel Density Estimation

# Crie uma nova coluna 'EstacaoAno' com a estação do ano correspondente para cada data
stage['EstacaoAno'] = stage['Data'].dt.month.map({1: 'Verão', 2: 'Verão', 3: 'Outono', 4: 'Outono', 5: 'Outono', 6: 'Inverno', 7: 'Inverno', 8: 'Inverno', 9: 'Primavera', 10: 'Primavera', 11: 'Primavera', 12: 'Verão'})

# Crie um DataFrame separado para os valores máximos diários de cotas por estação do ano
stage_max_por_estacao = stage.groupby(['EstacaoAno', stage['Data'].dt.year])['Cota'].max().reset_index()

# Renomeie as colunas conforme necessário
stage_max_por_estacao.columns = ['EstacaoAno', 'Ano', 'MaxCotasDiario']

# Exibe o novo DataFrame
print(stage_max_por_estacao)

# Função kdeplot da biblioteca Seaborn para criar um gráfico de densidade de kernel
plt.figure(figsize=(10, 6))
sns.kdeplot(data=stage_max_por_estacao, x='MaxCotasDiario', hue='EstacaoAno', fill=True)
plt.title('Densidade de Kernel dos Valores Máximos Diários de Cotas por Estação do Ano')
plt.xlabel('MaxCotasDiario')
plt.ylabel('Densidade')
plt.show()

#%% Cálculo do tempo de retorno

df_tr = pd.DataFrame({'Cota': sorted(list_cotas_maximas, reverse=True)}) # ordena a lista de cotas máximas em ordem decrescente
df_tr['TREmpirico'] = (len(list_cotas_maximas)+1)/(df_tr.index+1)

# Escolha uma colormap (mapa de cores)
cmap = cm.get_cmap('hot_r') # o '_r' é para inverter a escala de cores
#cmap = cm.get_cmap('viridis_r')

# Gráfico
plot = plt.scatter(df_tr['TREmpirico'], df_tr['Cota'], marker='o', c=df_tr['Cota'], cmap=cmap)
plt.xscale('log')  # Use escala logarítmica no eixo x para representar tempos de retorno
plt.xlabel('Tempo de Retorno Empírico (anos)')
plt.ylabel('Cota Máxima Anual (cm)')
plt.title('Cotas Máximas Anuais em Manaus')

# Adicione linha horizontal
#plt.axhline(y=2997, color='red', linestyle='--', label='Cota máxima de 2012: 2997 cm')
plt.axhline(y=2997, color='red', linestyle='--')

plt.colorbar(plot, label='Cotas Máximas')  # Adiciona a barra de cor
plt.legend() 

plt.grid(True)
plt.show()

# Três registros históricos em Manaus: 2009, 2012 e 2021

#%% Ajuste da Distribuição de Gumbel

x = np.linspace(df_cotas_maximas['Cota'].min(), 1.15*df_cotas_maximas['Cota'].max(), 1000)
params_gumbel = gumbel_r.fit(df_cotas_maximas['Cota'])
loc_gumbel, scale_gumbel = params_gumbel # estimativa pelo Método da Máxima Verossimilhança não ficou bom

# A saída do python para o scale_gumbel está dando o dobro

# Parâmetros da distribuição Gumbel pelo método dos momentos
media = np.mean(list_cotas_maximas)
desvpad = np.std(list_cotas_maximas, ddof=1) #ddof=1 indica que é o cálculo do desvio padrão amostral
mu = media - 0.45*desvpad # parâmetro de posição
beta = 0.7797*desvpad # parâmetro de escala

# Visualização do Ajuste da Distribuição
#pdf_fitted_gumbel = gumbel_r.pdf(x, loc=loc_gumbel, scale=scale_gumbel) # pdf está muito "plana", pois scale grande indica uma grande dispersão
pdf_fitted_gumbel = gumbel_r.pdf(x, loc=2724.22, scale=98.53)

plt.hist(df_cotas_maximas['Cota'], bins=6, density=True, alpha=0.6, color='blue')
plt.plot(x, pdf_fitted_gumbel, 'r-', label='Gumbel Fit')
plt.axvline(x=2997, color='red', linestyle='--')
plt.title('Ajuste da Distribuição de Gumbel')
plt.legend()
plt.show()

# Cálculo do Quantil para Tempo de Retorno de 100 anos (ou outro valor desejado)
tr_100 = 100
quantil_tr_100_gumbel = gumbel_r.ppf(1 - 1/tr_100, loc=loc_gumbel, scale=scale_gumbel)

# Imprima o quantil para o tempo de retorno desejado
print(f'Quantil para Tempo de Retorno de {tr_100} anos (Gumbel): {quantil_tr_100_gumbel}')

## TESTE: cálculo do TR a partir do valor de uma cota máxima anual em uma
# distribuição de Gumbel pode ser realizado usando a função de confiabilidade inversa da distribuição Gumbel

# Suponha que você tenha uma cota máxima anual específica
cota_maxima_anual = 2997

# Defina a função de confiabilidade para a distribuição Gumbel
def reliability_function(x):
    return np.exp(-np.exp(-(x - mu) / beta))

# Calcule a função de confiabilidade para a cota máxima anual
F_x = reliability_function(cota_maxima_anual)

# Calcule o Tempo de Retorno
tempo_retorno = 1 / (1 - F_x)

print(f'Tempo de Retorno para a cota máxima anual de {cota_maxima_anual}: {tempo_retorno:.2f} anos')


# Plote das curvas de frequência
plt.figure(figsize=(8, 5))
plt.plot(df_tr['TREmpirico'], df_tr['Cota'], label='Dados Empíricos', marker='o')
plt.plot(tempo_retorno_sintetico, dados_sinteticos, label='Gumbel Estimado', linestyle='--')

plt.title('Curvas de Frequência: Dados Empíricos vs. Gumbel Estimado')
plt.xlabel('Tempo de Retorno')
plt.ylabel('Cotas Máximas Anuais')
plt.xscale('log')  # Use escala logarítmica no eixo x para representar tempos de retorno
plt.legend()
plt.grid(True)
plt.show()



#%% Ajuste da Distribuição Lognormal

params = lognorm.fit(df_cotas_maximas['Cota'])
shape, loc, scale = params

# Cálculo do Quantil para Tempo de Retorno de 100 anos (ou outro valor desejado)
tr_100 = 100  # Tempo de Retorno em anos
quantil_tr_100 = lognorm.ppf(1 - 1/tr_100, shape, loc=loc, scale=scale)

# Visualização do Ajuste da Distribuição
x = np.linspace(df_cotas_maximas['Cota'].min(), df_cotas_maximas['Cota'].max(), 1000)
pdf_fitted = lognorm.pdf(x, shape, loc=loc, scale=scale)

plt.hist(df_cotas_maximas['Cota'], bins=30, density=True, alpha=0.6, color='g')
plt.plot(x, pdf_fitted, 'r-', label='Lognormal Fit')
plt.title('Ajuste da Distribuição Lognormal')
plt.legend()
plt.show() 

# Imprima o quantil para o tempo de retorno desejado
print(f'Quantil para Tempo de Retorno de {tr_100} anos: {quantil_tr_100}')






