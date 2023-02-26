from time import time
from typing import List, Optional

import lightning.lightning_app
from lightning.lightning_app.core.queues import QueuingSystem
from lightning.lightning_app.runners.backends.backend import Backend


class DockerBackend(Backend):
    def resolve_url(self, app, base_url: Optional[str] = None) -> None:
        pass

    def stop_work(self, app: "lightning.lightning_app.LightningApp", work: "lightning.lightning_app.LightningWork") -> None:
        pass

    def __init__(self, entrypoint_file: str):
        super().__init__(entrypoint_file=entrypoint_file, queues=QueuingSystem.REDIS, queue_id=str(int(time())))

    def create_work(self, app, work):
        pass

    def update_work_statuses(self, works) -> None:
        pass

    def stop_all_works(self, works: List["lightning.lightning_app.LightningWork"]) -> None:
        pass
