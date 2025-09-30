FROM python:3.10-slim


# system deps for OpenCV and optional libs
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential cmake libglib2.0-0 libsm6 libxext6 libxrender-dev \
&& apt-get clean && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


COPY . /app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"
