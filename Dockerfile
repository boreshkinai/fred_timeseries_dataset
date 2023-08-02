FROM nvidia/cuda:11.1.1-cudnn8-devel

ENV PROJECT_PATH /workspace/fred_timeseries_dataset

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN date
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8 && apt-get -y install git g++ zip unzip gnupg software-properties-common curl wget
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

RUN echo "**** Installing Python ****" && \
    apt-get update && \
    add-apt-repository 'ppa:deadsnakes/ppa' &&  \
    apt-get install -y build-essential python3.7 python3.7-dev python3-pip python3.7-distutils && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.7 get-pip.py && \
    rm -rf /var/lib/apt/lists/*

# Install tini, which will keep the container up as a PID 1
RUN apt-get update && apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=0.19.0 && \
    curl -L "https://github.com/krallin/tini/releases/download/v0.19.0/tini_0.19.0.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
COPY ./requirements.txt ./requirements.txt
RUN python3.7 -m pip install install -r ./requirements.txt --ignore-installed PyYAML
    
RUN mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py 

ENTRYPOINT [ "/usr/bin/tini", "--" ]

CMD ["jupyter", "notebook", "--allow-root"]

WORKDIR ${PROJECT_PATH}
