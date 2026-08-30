from typing import Any

from surrealdb import AsyncSurreal

from .config import Settings


class Database:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Any = None

    async def connect(self) -> None:
        self.client = AsyncSurreal(self.settings.surreal_url)
        await self.client.connect()
        await self.client.signin(
            {
                "username": self.settings.surreal_user,
                "password": self.settings.surreal_pass,
            }
        )
        await self.client.use(
            self.settings.surreal_namespace,
            self.settings.surreal_database,
        )

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def version(self) -> str:
        self._require_connection()
        return str(await self.client.version())

    async def query(
        self,
        statement: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        self._require_connection()
        return await self.client.query(statement, variables or {})

    async def upsert(self, record: Any, data: dict[str, Any]) -> Any:
        self._require_connection()
        return await self.client.upsert(record, data)

    def _require_connection(self) -> None:
        if self.client is None:
            raise RuntimeError("SurrealDB is not connected")

