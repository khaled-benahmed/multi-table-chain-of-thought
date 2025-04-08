# Two-stage build produces slimer docker images.
FROM python:3.12-bullseye as builder
RUN pip install poetry==1.7.1
# Set Poetry environment variables for non-interactive installation
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache
WORKDIR /app
COPY pyproject.toml poetry.lock ./
# Install runtime dependencies only (excludes development dependencies)
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image
FROM python:3.12-slim-bullseye as runtime
# Set the virtual environment path
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"
# Copy the virtual environment from the builder stage
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
# Copy application code to the Docker image
COPY ./app /app
CMD ["python", "-m", "app.client"]