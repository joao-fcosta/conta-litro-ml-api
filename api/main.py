from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from enum import Enum
import joblib
import pandas as pd
import logging

# Configuração de Logs para monitorar erros em produção
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tenta carregar o modelo de ML
try:
    modelo = joblib.load("modelo_combustivel.joblib")
    logger.info("Modelo carregado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo: {e}")
    modelo = None

app = FastAPI(
    title="API de Previsão de Consumo de Combustível",
    description="Interface de predição para o projeto Conta Litro.",
    version="1.1.0"
)

# Enums para restringir entradas e evitar erros no OneHotEncoder
class FuelType(str, Enum):
    Z = "Z" # Premium
    X = "X" # Regular
    D = "D" # Diesel
    E = "E" # Ethanol

class FuelRequest(BaseModel):
    year: int = Field(..., gt=1900, lt=2030, description="Ano do veículo")
    make: str = Field(..., example="TOYOTA")
    model: str = Field(..., example="COROLLA")
    enginesize: float = Field(..., gt=0, example=2.0)
    cylinders: int = Field(..., gt=0, example=4)
    vehicleclass: str = Field(..., example="COMPACT")
    fuel: FuelType
    distance_km: float | None = Field(None, gt=0)

class FuelResponse(BaseModel):
    consumo_l_100km: float
    consumo_litros_viagem: float | None
    km_por_litro: float | None

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Verifica se o modelo está carregado e a API operacional."""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo de ML não disponível.")
    return {"status": "online", "model_loaded": True}

@app.post("/predict", response_model=FuelResponse)
def predict_consumption(request: FuelRequest):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado no servidor.")

    try:
        # Preparação dos dados para o modelo (mantendo as colunas do treino)
        input_data = pd.DataFrame([{
            "YEAR": request.year,
            "MAKE": request.make.upper(), # Padronização para maiúsculas
            "MODEL": request.model.upper(),
            "ENGINE SIZE": request.enginesize,
            "CYLINDERS": request.cylinders,
            "VEHICLE CLASS": request.vehicleclass.upper(),
            "FUEL": request.fuel.value
        }])

        # Predição
        pred_l_100km = float(modelo.predict(input_data)[0])

        # Lógica de negócio (Pode ser movida para um service.py futuramente)
        consumo_viagem = None
        km_por_litro = None

        if pred_l_100km > 0:
            km_por_litro = round(100.0 / pred_l_100km, 2)
            if request.distance_km:
                consumo_viagem = round((request.distance_km * pred_l_100km) / 100.0, 2)

        return FuelResponse(
            consumo_l_100km=round(pred_l_100km, 2),
            consumo_litros_viagem=consumo_viagem,
            km_por_litro=km_por_litro
        )

    except Exception as e:
        logger.error(f"Erro durante a predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar a predição. Verifique os dados de entrada."
        )