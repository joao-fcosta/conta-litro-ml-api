# <p align="center">Conta Litro ML API</p>

<p align="center">
  <strong>Projeto Final - Disciplina de Inteligência Artificial</strong><br>
  Engenharia de Software | Unicatólica-TO
</p>

---
## 📋 Sobre o Projeto
Esta API utiliza modelos de Machine Learning para prever o consumo de combustível de veículos com base em características técnicas. O projeto abrange desde o processamento de dados e treinamento de modelos de regressão até a exposição de um serviço web escalável.

## 📊 O Modelo de Machine Learning
O coração desta aplicação é um **VotingRegressor**, um modelo de ensemble que combina as forças de três algoritmos distintos para maximizar a precisão:
1. **Ridge Regression:** Focado em regularização para evitar overfitting.
2. **RandomForestRegressor:** Excelente para capturar relações não-lineares.
3. **GradientBoostingRegressor:** Focado em reduzir o erro residual sequencialmente.

### Resultados Obtidos:
- **R² Score:** ~0.94 (O modelo explica 94% da variância dos dados).
- **RMSE:** ~0.86 L/100km.

## ⚙️ Tecnologias
- **Linguagem:** Python 3.14
- **Framework Web:** FastAPI (Uvicorn)
- **Data Science:** Pandas, Numpy, Scikit-Learn
- **Serialização:** Joblib

## 🛠️ Estrutura do Projeto
```text
conta-litro-ml-api/
├── api/
│   └── main.py          # Implementação dos endpoints FastAPI
├── data/
│   └── Fuel_Consumption.csv  # Dataset de treinamento
├── main.ipynb           # Notebook de treino e análise (Google Colab)
├── modelo_combustivel.joblib # Modelo treinado e serializado
└── requirements.txt     # Dependências do projeto
```

## 🛠️ Como Rodar o Projeto

### 1. Clonar o repositório e preparar o ambiente
```bash
git clone [https://github.com/joao-fcosta/conta-litro-ml-api.git](https://github.com/joao-fcosta/conta-litro-ml-api.git)
cd ContaLitro-Brain
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Gerar o Modelo de IA
```bash
python train_model.py
# Isso criará o arquivo modelo_combustivel.joblib na raiz do projeto.
```

### 3. Rodar a API
```bash
uvicorn main:app --reload
```
