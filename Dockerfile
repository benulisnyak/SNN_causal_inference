FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install the lightweight package first so dependency layers are reusable when
# only example data or documentation changes.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

COPY networks ./networks
COPY examples ./examples

ENTRYPOINT ["snn-connectivity"]
CMD ["--help"]
