FROM python:3.12-slim-bookworm

ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace \
    HOME=/home/paper \
    GRB_LICENSE_FILE=/licenses/gurobi.lic

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
        procps \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/paper.lock.txt /tmp/paper.lock.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/paper.lock.txt

RUN python -m ipykernel install --sys-prefix --name paper-kernel --display-name "Python (paper)"

COPY docker /opt/paper/docker

RUN set -eux; \
    mkdir -p /workspace /licenses /home/paper /workspace/tmp/jupyter-notebook; \
    if ! getent group "${GID}" >/dev/null; then groupadd --gid "${GID}" paper; fi; \
    if ! getent passwd "${UID}" >/dev/null; then useradd --uid "${UID}" --gid "${GID}" --create-home --home-dir /home/paper --shell /bin/bash paper; fi; \
    chown -R "${UID}:${GID}" /opt/paper /workspace /licenses /home/paper; \
    chmod +x /opt/paper/docker/*.sh

WORKDIR /workspace
USER ${UID}:${GID}

ENTRYPOINT ["tini", "--"]
CMD ["bash"]

