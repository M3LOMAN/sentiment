FROM python

RUN python -m pip install numpy tensorflow pandas nltk

WORKDIR /C:/Users/Владимир/OneDrive/Документы/sent_analysis

COPY . .

ENTRYPOINT ["python"]

CMD ["full_sentiment.py"]