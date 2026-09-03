# refs:
# - docs: https://fastapi.tiangolo.com/#alternative-api-docs
# - installation guide: https://fastapi.tiangolo.com/tutorial/#install-fastapi
# - include_router (splitting routes across files): https://fastapi.tiangolo.com/tutorial/bigger-applications/#include-the-apirouter
# - logging basicConfig: https://docs.python.org/3/library/logging.html#logging.basicConfig
import logging

from app.api.exception_handlers import register_exception_handlers
from app.api.routes.research import router as research_router
from fastapi import FastAPI

# Configure root logger once at import time so every module's
# `logging.getLogger(__name__)` call gets a usable handler. The
# `force=False` default (explicit here for clarity) means re-imports in
# tests don't reconfigure logging on every collection, and the
# `handlers` check below guards the same thing defensively in case
# something else (uvicorn, an IDE test runner) already attached one.
#
# NOTE: when running under `uvicorn` the server installs its own log
# handlers, and this basicConfig call becomes a no-op for the request
# log lines (uvicorn owns those). It still applies to anything our own
# code logs via getLogger(__name__) — exception handlers, service
# warnings, etc.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# You can ran the application using `uv run fastapi dev`
# server runs at http://127.0.0.1:8000 and Swagger documentation at http://127.0.0.1:8000/docs
app = FastAPI()

# Must run before include_routere because exception handlers need to be registered
# on the app before any request comes in, but registration itself doesn't
# depend on route registration order, this ordering is just for readability.
register_exception_handlers(app)

# The /ask route itself lives in api/routes/research.py, wired to
# domain/research/service.py. This file's only job is creating the app
# and mounting routers onto it, no route logic here.
app.include_router(research_router, tags=["research"])
