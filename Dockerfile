# Define function directory
ARG FUNCTION_DIR="/function"

FROM osgeo/gdal:ubuntu-small-3.6.2 as build-image

# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
  apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev \
  python3-pip

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Create function directory
RUN mkdir -p ${FUNCTION_DIR}

# Copy function code
COPY src/index_safe/* ${FUNCTION_DIR}/
COPY ./requirements_for_lambda.txt .

# Install the runtime interface client
RUN pip install \
        --target ${FUNCTION_DIR} \
        -r requirements_for_lambda.txt

# Multi-stage build: grab a fresh copy of the base image
FROM osgeo/gdal:ubuntu-small-3.6.2

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the build image dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}/

ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD [ "app.handler" ]
