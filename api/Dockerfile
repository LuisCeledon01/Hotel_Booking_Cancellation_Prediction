FROM python:3.12-slim

# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

# libgomp1: lo necesita xgboost (no viene en la imagen slim).
# build-essential: por si alguna dependencia debe compilarse.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /opt/hotel-api

# Instalar dependencias y el modelo empaquetado (wheel)
ADD ./hotel-api /opt/hotel-api/
RUN pip install --upgrade pip
RUN pip install -r /opt/hotel-api/requirements.txt
RUN pip install /opt/hotel-api/model-pkg/*.whl

# Hacer el directorio de trabajo ejecutable
RUN chmod +x /opt/hotel-api/run.sh
# Cambiar propiedad de la carpeta a api-user
RUN chown -R api-user:api-user ./

USER api-user
# Puerto a exponer para la api
EXPOSE 8001

# Comandos a ejecutar al correr el contenedor
CMD ["bash", "./run.sh"]
