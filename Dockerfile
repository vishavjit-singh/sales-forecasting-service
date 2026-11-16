# Forecast API image. The SYNTHETIC data is generated, backtested and trained at build time with the same
# commands `make demo` runs, so the image is self-contained and reproducible.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml ./
COPY src ./src
RUN python -m salesforecast generate && python -m salesforecast backtest && python -m salesforecast train

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "salesforecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
