"""FastAPI 의존성."""

from interfaces.http.dependencies.container import container_dependency, get_container

__all__ = ["container_dependency", "get_container"]
