FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get -y install sudo

RUN echo "==> Installing Python3 and pip ...." \  
 && apt-get install python3 -y \
 && apt install python3-pip -y \
 && apt install openssh-client -y

RUN echo "==> Adding Rich ..." \
 && pip install rich

RUN echo "==> Adding pyATS ..." \
 && pip install pyats[full]

RUN echo "==> Install dos2unix..." \
  && sudo apt-get install dos2unix -y 

RUN echo "==> Install langchain requirements -langchain" \
  && pip install langchain

RUN echo "==> Install langchain requirements - jq" \  
  && pip install jq

RUN echo "==> Install langchain requirements - chromadb" \
  && pip install chromadb

RUN echo "==> Install langchain requirements - openai" \
  && pip install openai

RUN echo "==> Install langchain requirements - tiktoken" \
  && pip install tiktoken

RUN echo "==> Install langchain requirements - torch" \
  && pip install torch

RUN echo "==> Install langchain requirements - transformers" \
  && pip install transformers

RUN echo "==> Install langchain requirements - InstructorEmbedding" \
  && pip install InstructorEmbedding

RUN echo "==> Install langchain requirements - sentence transformers" \  
  && pip install sentence-transformers

# Install tshark
RUN apt-get update && apt-get install -y tshark

RUN echo "==> Adding pyshark ..." \
  && pip install pyshark

COPY /chat_with_catalyst /chat_with_catalyst/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]