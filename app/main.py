# refs:
# - docs: https://fastapi.tiangolo.com/#alternative-api-docs
# - installation guide: https://fastapi.tiangolo.com/tutorial/#install-fastapi
# - include_router (splitting routes across files): https://fastapi.tiangolo.com/tutorial/bigger-applications/#include-the-apirouter
from app.api.exception_handlers import register_exception_handlers
from app.api.routes.research import router as research_router
from fastapi import FastAPI

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
