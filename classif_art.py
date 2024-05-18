import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import requests
from bs4 import BeautifulSoup

# Загрузка данных
# Предполагается, что у вас есть CSV файл с колонками "title", "description" и "genre"
df = pd.read_csv('dataset_1.csv', encoding='cp1251', header=0, sep=';')
print(df.keys().tolist())

# Предварительная обработка текста
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('russian'))

def preprocess_text(text):
    # Удаление спецсимволов и чисел, оставляем кириллические и пробелы
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    # Токенизация
    tokens = word_tokenize(text, language='russian')
    # Удаление стоп-слов
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Объединение обратно в строку
    text = ' '.join(filtered_tokens)
    return text

# Применение предобработки к названиям и описаниям
df['text'] = df['title'] + ' ' + df['description']
df['text'] = df['text'].apply(preprocess_text)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['genre'], test_size=0.2, random_state=42)

# Создание и обучение модели
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

pipeline.fit(X_train, y_train)


# Использование модели для предсказания жанра нового видео
def predict_genre(title, description):
    text = preprocess_text(title + ' ' + description)
    return pipeline.predict([text])[0]

# Пример использования
def get_video_metadata(video_url):
    response = requests.get(video_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Извлечение метаданных
    title_tag = soup.find('meta', property='og:title')
    description_tag = soup.find('meta', property='og:description')

    # Проверка наличия метаданных и извлечение их содержимого
    title = title_tag['content'] if title_tag else 'Название не найдено'
    description = description_tag['content'] if description_tag else 'Описание не найдено'

    # Приведение к строковому виду
    title = str(title)
    description = str(description)

    return title, description


# Пример использования функции для URL видео на YouTube
video_url = 'https://www.youtube.com/watch?v=zhncBFDbwPk'
title, description = get_video_metadata(video_url)

print("Название видео:", title)
print("Описание видео:", description)
predicted_genre = predict_genre(title, description)
print(f"Название: {title},  Предсказанный жанр: {predicted_genre}")