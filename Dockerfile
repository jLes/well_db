# WellDB Docker Image
# Uses Python 3.14 base with Playwright installed

FROM python:3.14-bookworm

WORKDIR /app

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY well_db/ ./well_db/
COPY resources/ ./resources/

# Initialize uv project and install dependencies
RUN uv sync

# Install Playwright browsers and system dependencies
RUN uv run playwright install --with-deps chromium

# Create directory for data persistence
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# Default command: start API server
CMD ["uv", "run", "python", "-m", "well_db", "serve", "--host", "0.0.0.0", "--port", "8000"]
