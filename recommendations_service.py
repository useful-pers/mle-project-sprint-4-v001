import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI

logger = logging.getLogger("uvicorn.error")

FEATURES_STORE_URL = "http://127.0.0.1:8010"
EVENTS_STORE_URL = "http://127.0.0.1:8020"


def get_storage_options() -> dict:
    """
    Параметры для чтения parquet из S3 через pandas.
    """
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
    Базовый S3-префикс, где лежат файлы рекомендаций.
    """
    bucket = os.getenv("BUCKET")
    return f"s3://{bucket}/recsys/recommendations/"


def dedup_ids(ids: list[int]) -> list[int]:
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение.
    """
    seen = set()
    return [x for x in ids if not (x in seen or seen.add(x))]


class RecommendationsStore:
    """
    При старте сервиса грузим:
    - персональные офлайн-рекомендации (ALS)
    - дефолтные (топ-популярные)
    """

    def __init__(self):
        self._personal: pd.DataFrame | None = None
        self._default: pd.DataFrame | None = None
        self._stats = {
            "request_offline_personal": 0,
            "request_offline_default": 0,
            "request_online": 0,
            "request_blended": 0,
            "errors": 0,
        }

    def load(self, base_recs: str, storage_options: dict) -> None:
        """
        Загружает офлайн-рекомендации из S3 в память.
        """
        personal_path = base_recs + "personal_als.parquet"
        default_path = base_recs + "top_popular.parquet"

        logger.info("Loading personal offline recs from %s", personal_path)
        personal = pd.read_parquet(
            personal_path,
            storage_options=storage_options,
            columns=["user_id", "item_id", "rank"],
        )
        
        personal = personal.sort_values(["user_id", "rank"]).set_index("user_id")
        self._personal = personal
        logger.info("Loaded personal offline recs: rows=%s", len(personal))

        logger.info("Loading default offline recs from %s", default_path)
        default = pd.read_parquet(
            default_path,
            storage_options=storage_options,
            columns=["item_id", "rank"],
        )
        default = default.sort_values("rank")
        self._default = default
        logger.info("Loaded default offline recs: rows=%s", len(default))

    def get_offline(self, user_id: int, k: int = 100) -> list[int]:
        """
        Возвращает офлайн-рекомендации:
        - если user_id есть в personal_als → персональные
        - иначе → top_popular
        """
        try:
            if self._personal is None or self._default is None:
                return []

            try:
                recs = self._personal.loc[user_id]["item_id"].to_list()[:k]
                self._stats["request_offline_personal"] += 1
                return recs
            except KeyError:
                recs = self._default["item_id"].to_list()[:k]
                self._stats["request_offline_default"] += 1
                return recs

        except Exception:
            self._stats["errors"] += 1
            logger.exception("Offline recommendations error")
            return []

    def inc(self, key: str) -> None:
        if key in self._stats:
            self._stats[key] += 1

    def stats(self) -> dict:
        return dict(self._stats)


rec_store = RecommendationsStore()


def get_events(user_id: int, k: int) -> list[int]:
    """
    Забирает последние события пользователя из Event Store.
    """
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    params = {"user_id": user_id, "k": k}
    resp = requests.post(EVENTS_STORE_URL + "/get", headers=headers, params=params, timeout=3)
    resp.raise_for_status()
    return resp.json().get("events", [])


def get_similar_items(item_id: int, k: int) -> dict:
    """
    Забирает похожие айтемы из Feature Store.
    """
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    params = {"item_id": item_id, "k": k}
    resp = requests.post(FEATURES_STORE_URL + "/similar_items", headers=headers, params=params, timeout=3)
    resp.raise_for_status()
    return resp.json()


def get_online(user_id: int, k: int = 100, last_n_events: int = 3) -> list[int]:
    """
    Онлайн-рекомендации по последним last_n_events событиям:
    - берём события из Event Store
    - для каждого события берём похожие айтемы из Feature Store
    - объединяем, сортируем по score, дедуп, режем до k
    """
    try:
        events = get_events(user_id=user_id, k=last_n_events)
        if not events:
            return []

        items: list[int] = []
        scores: list[float] = []

        for event_item_id in events:
            sim = get_similar_items(item_id=event_item_id, k=k)
            # Feature Store возвращает {"item_id_2": [...], "score": [...]}
            items += sim.get("item_id_2", [])
            scores += sim.get("score", [])

        combined = list(zip(items, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        combined_ids = [item_id for item_id, _ in combined]

        recs = dedup_ids(combined_ids)
        return recs[:k]

    except Exception:
        rec_store.inc("errors")
        logger.exception("Online recommendations error")
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    При старте сервиса грузим офлайн-рекомендации из S3 в память.
    """
    logger.info("Starting recommendations_service")

    load_dotenv(find_dotenv(), override=True)
    storage_options = get_storage_options()
    base_recs = get_base_recs_path()

    rec_store.load(base_recs=base_recs, storage_options=storage_options)

    logger.info("recommendations_service ready")
    yield

    logger.info("Stats: %s", rec_store.stats())
    logger.info("Stopping recommendations_service")


app = FastAPI(title="recommendations", lifespan=lifespan)


@app.post("/health")
async def health():
    return {"status": "healthy"}


@app.post("/stats")
async def stats():
    return rec_store.stats()


@app.post("/recommendations_offline")
async def recommendations_offline(user_id: int, k: int = 100):
    """
    Офлайн-рекомендации (personal ALS -> fallback top popular).
    """
    recs = rec_store.get_offline(user_id=user_id, k=k)
    return {"recs": recs}


@app.post("/recommendations_online")
async def recommendations_online(user_id: int, k: int = 100):
    """
    Онлайн-рекомендации по последним событиям.
    """
    rec_store.inc("request_online")
    recs = get_online(user_id=user_id, k=k, last_n_events=3)
    return {"recs": recs}


@app.post("/recommendations")
async def recommendations(user_id: int, k: int = 100):
    """
    Смешанные рекомендации:
    - онлайн на позициях 1,3,5...
    - офлайн на позициях 2,4,6...
    - потом хвосты, дедуп, обрезка до k
    """
    rec_store.inc("request_blended")

    recs_offline = (await recommendations_offline(user_id, k))["recs"]
    recs_online = (await recommendations_online(user_id, k))["recs"]

    blended: list[int] = []

    min_len = min(len(recs_offline), len(recs_online))
    for i in range(min_len):
        blended.append(recs_online[i])
        blended.append(recs_offline[i])

    if len(recs_online) > min_len:
        blended.extend(recs_online[min_len:])
    if len(recs_offline) > min_len:
        blended.extend(recs_offline[min_len:])

    blended = dedup_ids(blended)
    blended = blended[:k]

    return {"recs": blended}