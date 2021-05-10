ARG WORK_DIR="/home/polyaxon"
ARG USER="polyaxon"

WORKDIR $WORK_DIR

# Add lines here to copy over your src folder and 
# any other files you need in the image (like the saved model).
COPY src src
COPY README.md conda.yml ./
COPY model.h5 src


# Add a line here to update the base conda environment using the conda.yml. 
COPY conda.yml $WORK_DIR
RUN conda env update --file conda.yml -n base && rm -f conda.yml 

# DO NOT remove the following line - it is required for deployment on Tekong
RUN chown -R 1000450000:0 $WORK_DIR

USER $USER

EXPOSE 8000

# Add a line here to run your app
ENTRYPOINT python -m src.app



