# Base Arguements to leverage across build stages
ARG APP_DIR=/app

#############
# Build image
#############
FROM python:3.10-slim-buster AS build
ARG APP_DIR

# Install curl so we can get poetry
# install build-base and libffi-dev so we can install poetry and dependencies (compiles some code)
RUN apt-get update && apt-get install -y \
  curl build-essential libffi-dev \
  && rm -rf /var/lib/apt/lists/*

# install poetry and add to path
ENV POETRY_VERSION=1.3.2
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH /root/.local/bin:$PATH

# change work directory for where our app will live
WORKDIR ${APP_DIR}
COPY pyproject.toml poetry.lock ./

# Copy over the venv including any symbolic links, do not install development/testing libraries when install poetry dependencies
RUN python -m venv --copies ${APP_DIR}/venv
RUN . ${APP_DIR}/venv/bin/activate && poetry install 

#############
# Deployment image
#############
FROM python:3.10-slim-buster as prod
ARG APP_DIR

COPY --from=build ${APP_DIR}/venv ${APP_DIR}/venv/
ENV PATH ${APP_DIR}/venv/bin:$PATH

WORKDIR ${APP_DIR}/
COPY . ./

HEALTHCHECK --start-period=30s CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# # Base Arguements to leverage across build stages
# ARG APP_DIR=/app

# # Base image
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base

# ARG APP_DIR

# # Install dependencies
# RUN apt-get update && \
#     apt-get install -y \
#         curl build-essential \
#         python3-pip \
#         python3-dev \
#         wget \
#         git && \
#     rm -rf /var/lib/apt/lists/*

# #install poetry
# ENV POETRY_version = 1.3.2
# RUN  curl -sSL https://install.python-poetry.org | python3 -
# ENV PATH /root/.local/bin:$PATH

# #prepare venv
# WORKDIR ${APP_DIR}
# COPY pyproject.toml poetry.lock ./

# RUN python3 -m venv --copies ${APP_DIR}/venv --without-pip
# RUN . ${APP_DIR}/venv/bin/activate && poetry install --no-root

# # Install PyTorch and torchvision
# # RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# FROM python:3.10-slim-bullseye as prod
# ARG APP_DIR

# RUN apt-get update && apt-get install -y \
#   curl build-essential libffi-dev \
#   && rm -rf /var/lib/apt/lists/*

# COPY --from=base ${APP_DIR}/venv ${APP_DIR}/venv/
# ENV PATH ${APP_DIR}/venv/bin:$PATH

# # Set environment variables
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# # copy src
# WORKDIR ${APP_DIR}/
# COPY . ./

# HEALTHCHECK --start-period=5s CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)"

# #run uvicorn
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
