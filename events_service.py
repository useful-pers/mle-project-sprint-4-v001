from fastapi import FastAPI


class EventStore:
    """
    Хранит события в оперативке (перезапуск сервиса = очистка истории)
    """

    def __init__(self, max_events_per_user: int = 10):
        # user_id -> список item_id (самое последнее событие первым)
        self._events: dict[int, list[int]] = {}
        self.max_events_per_user = max_events_per_user

    def put(self, user_id: int, item_id: int) -> None:
        """
        Добавляет событие (user_id, item_id) в начало списка событий пользователя.
        Длина истории ограничена max_events_per_user.
        """
        user_events = self._events.get(user_id, [])
        self._events[user_id] = [item_id] + user_events[: self.max_events_per_user]

    def get(self, user_id: int, k: int = 10) -> list[int]:
        """
        Возвращает последние k событий пользователя (последние = первые в списке).
        """
        return self._events.get(user_id, [])[:k]


events_store = EventStore()

app = FastAPI(title="events")


@app.post("/put")
async def put(user_id: int, item_id: int):
    """
    Сохраняет событие для пользователя: user_id взаимодействовал с item_id.
    """
    events_store.put(user_id, item_id)
    return {"result": "ok"}


@app.post("/get")
async def get(user_id: int, k: int = 10):
    """
    Возвращает список последних k событий пользователя.
    """
    return {"events": events_store.get(user_id, k)}