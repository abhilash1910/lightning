from time import time
from typing import List, Optional

import lightning.app
from lightning.app.core.queues import QueuingSystem
from lightning.app.runners.backends.backend import Backend


class DockerBackend(Backend):
    def resolve_url(self, app, base_url: Optional[str] = None) -> None:
        pass

    def stop_work(self, app: "lightning.app.LightningApp", work: "lightning.app.LightningWork") -> None:
        pass

    def __init__(self, entrypoint_file: str):
        super().__init__(entrypoint_file=entrypoint_file, queues=QueuingSystem.REDIS, queue_id=str(int(time())))

    def create_work(self, app, work):
        pass

    def update_work_statuses(self, works) -> None:
        pass

    def stop_all_works(self, works: List["lightning.app.LightningWork"]) -> None:
        pass
