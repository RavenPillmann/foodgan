FROM jjanzic/docker-python3-opencv

WORKDIR /app

COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME Serve

CMD ["python", "model_server.py", "--model", "all_ingredient_gan_gen_biggest_52.h5"]
