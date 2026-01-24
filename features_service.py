import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI

# Используем логгер uvicorn, чтобы сообщения шли в общий поток логов сервиса
logger = logging.getLogger("uvicorn.error")


def get_storage_options() -> dict:
    """
    Формирует параметры доступа к S3.
    """
    # Загружаем переменные окружения из .env
    load_dotenv(find_dotenv(), override=True)

    return {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {
            "endpoint_url": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
            "region_name": os.getenv("AWS_DEFAULT_REGION"),
        },
    }


def get_base_recs_path() -> str:
    """
    Возвращает базовый путь в S3, где лежат все артефакты рекомендаций.
    """
    bucket = os.getenv("BUCKET")
    return f"s3://{bucket}/recsys/recommendations/"


class SimilarItems:
    """
    In-memory хранилище item2item похожестей.

    При старте сервиса весь parquet загружается в память,
    чтобы дальше отвечать на запросы без походов в S3.
    """

    def __init__(self):
        # DataFrame с индексом item_id и колонками:
        # similar_item_id, score
        self._df: pd.DataFrame | None = None

    def load(self, path: str, storage_options: dict) -> None:
        """
        Загружает parquet с похожими объектами из S3 в память.
        Вызывается один раз при старте сервиса.
        """
        logger.info("Loading similar items from %s", path)

        df = pd.read_parquet(
            path,
            storage_options=storage_options,
            columns=["item_id", "similar_item_id", "score"],
        )

        df = df.set_index("item_id").sort_index()

        self._df = df

        logger.info("Loaded similar items: rows=%s", len(df))

    def get(self, item_id: int, k: int = 10) -> dict:
        """
        Возвращает до k объектов, похожих на item_id.

        """
        # Если данные ещё не загружены 
        if self._df is None:
            return {"item_id_2": [], "score": []}

        try:
            # Берём первые k похожих объектов для данного item_id
            part = self._df.loc[item_id].head(k)

            return {
                "item_id_2": part["similar_item_id"].tolist(),
                "score": part["score"].astype(float).tolist(),
            }

        # Если для item_id нет похожих объектов
        except KeyError:
            logger.info("No similar items for item_id=%s", item_id)
            return {"item_id_2": [], "score": []}

        # Любая другая ошибка — логируем и возвращаем пустой результат
        except Exception:
            logger.exception("Failed to get similar items for item_id=%s", item_id)
            return {"item_id_2": [], "score": []}


sim_items_store = SimilarItems()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Жизненный цикл FastAPI-приложения.

    Здесь выполняется загрузка данных в память при старте сервиса.
    """
    logger.info("Starting features_service")

    storage_options = get_storage_options()
    base_recs = get_base_recs_path()

    # Путь к parquet с похожими объектами в S3
    path = base_recs + "similar.parquet"

    sim_items_store.load(path=path, storage_options=storage_options)

    logger.info("features_service ready")
    yield

    logger.info("Stopping features_service")


app = FastAPI(title="features", lifespan=lifespan)


@app.post("/similar_items")
async def similar_items(item_id: int, k: int = 10):
    """
    Endpoint Feature Store.

    Возвращает список объектов, похожих на item_id,
    используется сервисом рекомендаций для онлайн-рекомендаций.
    """
    return sim_items_store.get(item_id=item_id, k=k)