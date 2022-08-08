FROM python:3.8-bullseye

# Install dependencies
WORKDIR /build
COPY requirements.txt /build/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install tensorflow==2.9.0

# Configurable ENV variables
ENV PORT=8888
ENV DIR=/app 

WORKDIR ${DIR}

COPY . ${DIR}

COPY entrypoint.sh /build/entrypoint.sh
RUN chmod +x /build/entrypoint.sh
ENTRYPOINT /build/entrypoint.sh --port=${PORT}
