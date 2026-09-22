FROM spark:3.5.9-scala2.12-java17-python3-ubuntu

USER root
RUN python3 -m pip install --no-cache-dir numpy==1.26.4
RUN python3 -m pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --no-cache-dir sentence-transformers==6.1.0 \
    && mkdir -p /opt/huggingface \
    && chown -R 185:185 /opt/huggingface
RUN python3 -m pip install --no-cache-dir mlflow==3.16.1 \
    && mkdir -p /mlflow/artifacts \
    && chown -R 185:185 /mlflow
USER 185
