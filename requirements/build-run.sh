#!/bin/bash

if [ -z "$IMAGETAG" ]; then
  IMAGETAG="latest"
fi

docker build --build-arg NB_USER=${USER} --build-arg NB_UID=${UID} \
  --build-arg NB_GID=`id -g $USER` --force-rm \
  -t ml-py-workshop:${IMAGETAG} .

# TODO: add a forcedrun env var to force re-running container
# based on latest build (useful when adding packages while developing)
# FORCEDRUN="false"

if docker ps -a | grep -q ml-py-jupyterlab; then
  echo "---- found existing container, starting it ----"
  docker start -i ml-py-jupyterlab
else
  echo "---- no container found, running for the first time ----"
  # expose both Jupyter Lab and mkdocs server ports
  docker run --init -it -p 127.0.0.1:8888:8888 \
  -p 127.0.0.1:8001:8001 \
  -v `dirname ${PWD}`:/home/${USER}/work \
  -v /home/${USER}/.ssh:/home/${USER}/.ssh:ro \
  --name ml-py-jupyterlab ml-py-workshop:${IMAGETAG}
fi
