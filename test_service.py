import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv, find_dotenv


RECS_URL = "http://127.0.0.1:8000"
EVENTS_URL = "http://127.0.0.1:8020"
LOG_PATH = "test_service.log"

HEADERS = {"Content-type": "application/json", "Accept": "text/plain"}


def get_storage_options() -> dict:
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
    bucket = os.getenv("BUCKET")
    return f"s3://{bucket}/recsys/recommendations/"


def pick_users_with_personal(n: int = 2) -> list[int]:
    storage_options = get_storage_options()
    base_recs = get_base_recs_path()
    path = base_recs + "personal_als.parquet"

    df = pd.read_parquet(path, storage_options=storage_options, columns=["user_id"])
    return df["user_id"].drop_duplicates().head(n).tolist()


def post_json(url: str, params: dict) -> dict:
    resp = requests.post(url, headers=HEADERS, params=params, timeout=5)
    if resp.status_code != 200:
        raise RuntimeError(f"Request failed: {url}, status={resp.status_code}, body={resp.text}")
    return resp.json()


def main():
    # Простой “двойной вывод”: и в консоль, и в файл.
    # Без tee, без редиректов, без магии shell.
    with open(LOG_PATH, "w", encoding="utf-8") as f:

        def log(msg: str = ""):
            print(msg)
            f.write(msg + "\n")

        log("Check services health")
        health = post_json(RECS_URL + "/health", params={})
        log(f"recommendations_service /health: {health}")
        log()

        users = pick_users_with_personal(n=2)
        if len(users) < 2:
            raise RuntimeError("Не удалось получить 2 user_id из personal_als.parquet")

        user_personal_no_online = users[0]
        user_personal_with_online = users[1]
        user_no_personal = -1
        k = 10

        log("CASE 1: user without personal recs (expect default/top_popular)")
        recs_offline = post_json(RECS_URL + "/recommendations_offline", params={"user_id": user_no_personal, "k": k})
        recs_blended = post_json(RECS_URL + "/recommendations", params={"user_id": user_no_personal, "k": k})
        log(f"user_id: {user_no_personal}")
        log(f"offline: {recs_offline}")
        log(f"blended: {recs_blended}")
        log()

        log("CASE 2: user with personal recs, but without online history")
        online = post_json(RECS_URL + "/recommendations_online", params={"user_id": user_personal_no_online, "k": k})
        offline = post_json(RECS_URL + "/recommendations_offline", params={"user_id": user_personal_no_online, "k": k})
        blended = post_json(RECS_URL + "/recommendations", params={"user_id": user_personal_no_online, "k": k})
        log(f"user_id: {user_personal_no_online}")
        log(f"online: {online}")
        log(f"offline: {offline}")
        log(f"blended: {blended}")
        log()

        log("CASE 3: user with personal recs and online history")
        seed_items = offline["recs"][:4]
        log(f"Put events (item_id) into Event Store: {seed_items}")

        for item_id in seed_items:
            post_json(EVENTS_URL + "/put", params={"user_id": user_personal_with_online, "item_id": item_id})

        time.sleep(0.2)

        online3 = post_json(RECS_URL + "/recommendations_online", params={"user_id": user_personal_with_online, "k": k})
        offline3 = post_json(RECS_URL + "/recommendations_offline", params={"user_id": user_personal_with_online, "k": k})
        blended3 = post_json(RECS_URL + "/recommendations", params={"user_id": user_personal_with_online, "k": k})

        log(f"user_id: {user_personal_with_online}")
        log(f"online: {online3}")
        log(f"offline: {offline3}")
        log(f"blended: {blended3}")
        log()

        log("Done")
        log(f"Saved log to: {LOG_PATH}")


if __name__ == "__main__":
    main()