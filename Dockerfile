FROM python:3.12
WORKDIR /workspace

# 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && \
    apt-get install -y git vim curl && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /workspace/
RUN pip install -r  /workspace/requirements.txt

COPY ./ ./

ENTRYPOINT ["python", "server.py"]
