import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Melhor usar data_id direto, pra garantir que pega a versao certa
# aqui ele pega o balanced, que estão os numeros de 0 a 9 e o alfabeto maiusculo e minusculo,
# deixando algunas letras que são iguais tanto minuculo ou maiusculo
emnist = fetch_openml(data_id=41039, as_frame=False)

X = emnist.data      
y = emnist.target.astype(int)  # classes de 0 a 46


# Pegou as amostras e vai pegar 20% de teste e 80% para treino
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Classificador Binário se é Verdadeiro ou Falso
classe_alvo = 5  #Se é 5 ou não

y_train_bin = (y_train_full == classe_alvo)
y_test_bin = (y_test_full == classe_alvo)

X_train_bin = X_train_full
X_test_bin = X_test_full


# Classificador multiclasse de 1 a 5
# a mascara serve para pegar as imagens que que se encaixem nas classes dos números,
# se eles serão utilizados ou não
mask_train = np.isin(y_train_full, [1, 2, 3, 4, 5])
mask_test = np.isin(y_test_full, [1, 2, 3, 4, 5])

X_train_digitos = X_train_full[mask_train]
y_train_digitos = y_train_full[mask_train]

X_test_digitos = X_test_full[mask_test]
y_test_digitos = y_test_full[mask_test]


# Classificador multiclasse para letras de A a E

letras_A_E = [10, 11, 12, 13, 14]  # é representado assim no emnist. A, B, C, D, E

# mesma coisa dos números, para selecionar apenas as letras, 
# mas aqui ele transforma em números de 0 a 4, objetivo de facilitar o treinamento do classificador
mask_train = np.isin(y_train_full, letras_A_E)
mask_test = np.isin(y_test_full, letras_A_E)

X_train_letras = X_train_full[mask_train]
y_train_letras = y_train_full[mask_train] - 10  # reindexar pra 0-4

X_test_letras = X_test_full[mask_test]
y_test_letras = y_test_full[mask_test] - 10



# Fazer teste e treino com os modelos
# Dependendo do modelo teria que normalizar os pixels que são sensiveis a escala
# é chamado de normaliização Min-Max para [0,1]

# normalizacão para o binário
X_train_bin = X_train_bin / 255.0
X_test_bin = X_test_bin / 255.0

# para os numeros 
X_train_digitos = X_train_digitos / 255.0
X_test_digitos = X_test_digitos / 255.0

# para as letras
X_train_letras = X_train_letras / 255.0
X_test_letras = X_test_letras / 255.0


#Modelo binário
# Random Forest
print("rodando randiom forest")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_bin, y_train_bin)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_bin, y_train_bin)

# Regressao Logistica
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_bin, y_train_bin)

# SVM
svm = SVC(random_state=42)
svm.fit(X_train_bin, y_train_bin)

# MLP Classifier
mlp = MLPClassifier(max_iter=300, random_state=42)
mlp.fit(X_train_bin, y_train_bin)
print("rodou todos os modelos")


#Modeo numero
rf_digitos = RandomForestClassifier(random_state=42)
rf_digitos.fit(X_train_digitos, y_train_digitos)

nb_digitos = GaussianNB()
nb_digitos.fit(X_train_digitos, y_train_digitos)

lr_digitos = LogisticRegression(max_iter=1000, random_state=42)
lr_digitos.fit(X_train_digitos, y_train_digitos)

svm_digitos = SVC(random_state=42)
svm_digitos.fit(X_train_digitos, y_train_digitos)

mlp_digitos = MLPClassifier(max_iter=300, random_state=42)
mlp_digitos.fit(X_train_digitos, y_train_digitos)


#Modelos letras
rf_letras = RandomForestClassifier(random_state=42)
rf_letras.fit(X_train_letras, y_train_letras)

nb_letras = GaussianNB()
nb_letras.fit(X_train_letras, y_train_letras)

lr_letras = LogisticRegression(max_iter=1000, random_state=42)
lr_letras.fit(X_train_letras, y_train_letras)

svm_letras = SVC(random_state=42)
svm_letras.fit(X_train_letras, y_train_letras)

mlp_letras = MLPClassifier(max_iter=300, random_state=42)
mlp_letras.fit(X_train_letras, y_train_letras)


# Predict dos modelos testes

# binario
y_pred_rf = rf.predict(X_test_bin)
y_pred_nb = nb.predict(X_test_bin)
y_pred_lr = lr.predict(X_test_bin)
y_pred_svm = svm.predict(X_test_bin)
y_pred_mlp = mlp.predict(X_test_bin)

# numero
y_pred_rf_d = rf_digitos.predict(X_test_digitos)
y_pred_nb_d = nb_digitos.predict(X_test_digitos)
y_pred_lr_d = lr_digitos.predict(X_test_digitos)
y_pred_svm_d = svm_digitos.predict(X_test_digitos)
y_pred_mlp_d = mlp_digitos.predict(X_test_digitos)

# letras
y_pred_rf_l = rf_letras.predict(X_test_letras)
y_pred_nb_l = nb_letras.predict(X_test_letras)
y_pred_lr_l = lr_letras.predict(X_test_letras)
y_pred_svm_l = svm_letras.predict(X_test_letras)
y_pred_mlp_l = mlp_letras.predict(X_test_letras)



#Gerar métricas
# binário
report_rf = classification_report(y_test_bin, y_pred_rf, output_dict=True)
report_nb = classification_report(y_test_bin, y_pred_nb, output_dict=True)
report_lr = classification_report(y_test_bin, y_pred_lr, output_dict=True)
report_svm = classification_report(y_test_bin, y_pred_svm, output_dict=True)
report_mlp = classification_report(y_test_bin, y_pred_mlp, output_dict=True)
#converter em dataframe
df_rf = pd.DataFrame(report_rf).transpose()
df_nb = pd.DataFrame(report_nb).transpose()
df_lr = pd.DataFrame(report_lr).transpose()
df_svm = pd.DataFrame(report_svm).transpose()
df_mlp = pd.DataFrame(report_mlp).transpose()

#numeros
report_rf_d = classification_report(y_test_digitos, y_pred_rf_d, output_dict=True)
report_nb_d = classification_report(y_test_digitos, y_pred_nb_d, output_dict=True)
report_lr_d = classification_report(y_test_digitos, y_pred_lr_d, output_dict=True)
report_svm_d = classification_report(y_test_digitos, y_pred_svm_d, output_dict=True)
report_mlp_d = classification_report(y_test_digitos, y_pred_mlp_d, output_dict=True)
#dataframe
df_rf_d = pd.DataFrame(report_rf_d).transpose()
df_nb_d = pd.DataFrame(report_nb_d).transpose()
df_lr_d = pd.DataFrame(report_lr_d).transpose()
df_svm_d = pd.DataFrame(report_svm_d).transpose()
df_mlp_d = pd.DataFrame(report_mlp_d).transpose()

# letras
report_rf_l = classification_report(y_test_letras, y_pred_rf_l, output_dict=True)
report_nb_l = classification_report(y_test_letras, y_pred_nb_l, output_dict=True)
report_lr_l = classification_report(y_test_letras, y_pred_lr_l, output_dict=True)
report_svm_l = classification_report(y_test_letras, y_pred_svm_l, output_dict=True)
report_mlp_l = classification_report(y_test_letras, y_pred_mlp_l, output_dict=True)
#dataframe
df_rf_l = pd.DataFrame(report_rf_l).transpose()
df_nb_l = pd.DataFrame(report_nb_l).transpose()
df_lr_l = pd.DataFrame(report_lr_l).transpose()
df_svm_l = pd.DataFrame(report_svm_l).transpose()
df_mlp_l = pd.DataFrame(report_mlp_l).transpose()

# binário
df_rf["modelo"] = "RandomForest"
df_nb["modelo"] = "NaiveBayes"
df_lr["modelo"] = "LogisticRegression"
df_svm["modelo"] = "SVM"
df_mlp["modelo"] = "MLPClassifier"

# digitos
df_rf_d["modelo"] = "RandomForest"
df_nb_d["modelo"] = "NaiveBayes"
df_lr_d["modelo"] = "LogisticRegression"
df_svm_d["modelo"] = "SVM"
df_mlp_d["modelo"] = "MLPClassifier"

# letras
df_rf_l["modelo"] = "RandomForest"
df_nb_l["modelo"] = "NaiveBayes"
df_lr_l["modelo"] = "LogisticRegression"
df_svm_l["modelo"] = "SVM"
df_mlp_l["modelo"] = "MLPClassifier"

# binário
df_final_bin = pd.concat([df_rf, df_nb, df_lr, df_svm, df_mlp])
df_final_bin.to_csv("resultados_binario.csv")

# digitos
df_final_digitos = pd.concat([df_rf_d, df_nb_d, df_lr_d, df_svm_d, df_mlp_d])
df_final_digitos.to_csv("resultados_digitos.csv")

# letras
df_final_letras = pd.concat([df_rf_l, df_nb_l, df_lr_l, df_svm_l, df_mlp_l])
df_final_letras.to_csv("resultados_letras.csv")


print("Calculando metricas de treino...")

# Predict no TREINO - binario
y_pred_rf_train = rf.predict(X_train_bin)
y_pred_nb_train = nb.predict(X_train_bin)
y_pred_lr_train = lr.predict(X_train_bin)
y_pred_svm_train = svm.predict(X_train_bin)
y_pred_mlp_train = mlp.predict(X_train_bin)

# Predict no TREINO - digitos
y_pred_rf_d_train = rf_digitos.predict(X_train_digitos)
y_pred_nb_d_train = nb_digitos.predict(X_train_digitos)
y_pred_lr_d_train = lr_digitos.predict(X_train_digitos)
y_pred_svm_d_train = svm_digitos.predict(X_train_digitos)
y_pred_mlp_d_train = mlp_digitos.predict(X_train_digitos)

# Predict no TREINO - letras
y_pred_rf_l_train = rf_letras.predict(X_train_letras)
y_pred_nb_l_train = nb_letras.predict(X_train_letras)
y_pred_lr_l_train = lr_letras.predict(X_train_letras)
y_pred_svm_l_train = svm_letras.predict(X_train_letras)
y_pred_mlp_l_train = mlp_letras.predict(X_train_letras)

# Reports de TREINO - binario
report_rf_train = classification_report(y_train_bin, y_pred_rf_train, output_dict=True)
report_nb_train = classification_report(y_train_bin, y_pred_nb_train, output_dict=True)
report_lr_train = classification_report(y_train_bin, y_pred_lr_train, output_dict=True)
report_svm_train = classification_report(y_train_bin, y_pred_svm_train, output_dict=True)
report_mlp_train = classification_report(y_train_bin, y_pred_mlp_train, output_dict=True)

# Reports de TREINO - digitos
report_rf_d_train = classification_report(y_train_digitos, y_pred_rf_d_train, output_dict=True)
report_nb_d_train = classification_report(y_train_digitos, y_pred_nb_d_train, output_dict=True)
report_lr_d_train = classification_report(y_train_digitos, y_pred_lr_d_train, output_dict=True)
report_svm_d_train = classification_report(y_train_digitos, y_pred_svm_d_train, output_dict=True)
report_mlp_d_train = classification_report(y_train_digitos, y_pred_mlp_d_train, output_dict=True)

# Reports de TREINO - letras
report_rf_l_train = classification_report(y_train_letras, y_pred_rf_l_train, output_dict=True)
report_nb_l_train = classification_report(y_train_letras, y_pred_nb_l_train, output_dict=True)
report_lr_l_train = classification_report(y_train_letras, y_pred_lr_l_train, output_dict=True)
report_svm_l_train = classification_report(y_train_letras, y_pred_svm_l_train, output_dict=True)
report_mlp_l_train = classification_report(y_train_letras, y_pred_mlp_l_train, output_dict=True)

print("Montando tabela de ranking...")

resumo = pd.DataFrame({
    "modelo": ["RandomForest", "NaiveBayes", "LogisticRegression", "SVM", "MLPClassifier"] * 3,
    "dataset": ["binario"]*5 + ["digitos"]*5 + ["letras"]*5,
    "acc_treino": [
        report_rf_train["accuracy"], report_nb_train["accuracy"], report_lr_train["accuracy"], report_svm_train["accuracy"], report_mlp_train["accuracy"],
        report_rf_d_train["accuracy"], report_nb_d_train["accuracy"], report_lr_d_train["accuracy"], report_svm_d_train["accuracy"], report_mlp_d_train["accuracy"],
        report_rf_l_train["accuracy"], report_nb_l_train["accuracy"], report_lr_l_train["accuracy"], report_svm_l_train["accuracy"], report_mlp_l_train["accuracy"],
    ],
    "acc_teste": [
        report_rf["accuracy"], report_nb["accuracy"], report_lr["accuracy"], report_svm["accuracy"], report_mlp["accuracy"],
        report_rf_d["accuracy"], report_nb_d["accuracy"], report_lr_d["accuracy"], report_svm_d["accuracy"], report_mlp_d["accuracy"],
        report_rf_l["accuracy"], report_nb_l["accuracy"], report_lr_l["accuracy"], report_svm_l["accuracy"], report_mlp_l["accuracy"],
    ],
})

resumo["diferenca"] = (resumo["acc_treino"] - resumo["acc_teste"]).abs()
resumo = resumo.sort_values(["dataset", "diferenca"])

print(resumo)
resumo.to_csv("resumo_ranking.csv", index=False)
print("resumo_ranking.csv gerado!")



# Salva os melhores modelos de cada dataset para uso no Streamlit

import joblib
import os

os.makedirs("modelos", exist_ok=True)

# o SVM teve o melhor equilibrio
# treino/teste nos 3 datasets 

joblib.dump(svm, "modelos/modelo_binario.pkl")
joblib.dump(svm_digitos, "modelos/modelo_digitos.pkl")
joblib.dump(svm_letras, "modelos/modelo_letras.pkl")

print("Modelos salvos na pasta 'modelos/'!")