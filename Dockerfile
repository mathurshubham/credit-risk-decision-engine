# ==========================================
# Stage 1: The Builder (Compiles dependencies)
# ==========================================
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
# --no-dev: Production only
# --frozen: Use the lock file strictly
RUN uv sync --frozen --no-dev --no-install-project

# ==========================================
# Stage 2: The Runtime (Slim & Secure)
# ==========================================
FROM python:3.10-slim-bookworm

WORKDIR /app

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Copy the environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Set environment path to use the virtualenv
ENV PATH="/app/.venv/bin:$PATH"

# --- FIX IS HERE ---
# Add /app/src to PYTHONPATH so Python can find 'project_alpha'
ENV PYTHONPATH="/app/src"

# Copy application code
COPY src/ ./src/
COPY model.joblib .

# Expose port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "project_alpha.app:app", "--host", "0.0.0.0", "--port", "8000"]