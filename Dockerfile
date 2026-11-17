FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY rlnav ./rlnav
RUN pip install --no-cache-dir ".[llm]"
RUN useradd -m runner
USER runner
# Default: simulator-free evaluation. For real CARLA run a CARLA server container and pass
# --backend carla with CARLA_HOST/CARLA_PORT env vars.
ENTRYPOINT ["python", "-m", "rlnav.evaluate"]
CMD ["--policy", "heuristic", "--episodes", "20"]
