FROM python:3.8

ARG NB_USER="user"
ARG NB_UID="1000"
ARG NB_GID="1000"

ENV SHELL=/bin/bash

RUN apt-get update && \
  apt-get install -y sudo && \
  apt-get install -y git python3-dev virtualenv && \
  useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
  chmod g+w /etc/passwd && \
  # give NB_USER passwordless sudo
  echo "${NB_USER}    ALL=(ALL)    NOPASSWD:    ALL" >> /etc/sudoers && \
  # Prevent apt-get cache from being persisted to this layer.
  rm -rf /var/lib/apt/lists/*

USER $NB_UID

WORKDIR "/home/${NB_USER}"

# Copy requirements, install and configure the kernel.
COPY --chown="${NB_UID}:${NB_GID}" "docker-requirements.txt" "/tmp/"
RUN virtualenv -p python3 venv && \
  venv/bin/pip install -r "/tmp/docker-requirements.txt"
  # venv/bin/pip install -r "/tmp/docker-requirements.txt" && \
  # TODO: move pyearth to requirement file when they fix the install
  # git clone -b issue191 git://github.com/scikit-learn-contrib/py-earth.git && \
  # cd py-earth && ../venv/bin/python setup.py build_ext --inplace --cythonize && \
  # cd .. && rm -rf py-earth

ENV PATH "/home/${NB_USER}/venv/bin:${PATH}"

CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0"]
