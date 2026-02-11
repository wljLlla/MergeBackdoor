FROM continuumio/miniconda3:latest
ADD . /app
WORKDIR /app
COPY environment.yaml /app/environment.yaml
RUN bash -c "\
    conda update -n base -c defaults conda -y && \
    conda env create -f environment.yaml && \
    conda clean -afy \
"
SHELL ["/bin/bash", "--login", "-c"]
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate merge3 && exec bash"]
