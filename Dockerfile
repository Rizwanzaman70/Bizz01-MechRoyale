FROM python:3.13-slim
WORKDIR /bot
COPY mechfight.py example_bot.py ./
ENTRYPOINT ["python3", "-u", "/bot/example_bot.py"]
