import requests
from bs4 import BeautifulSoup
import time
from transformers import pipeline
from langdetect import detect
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Загрузка модели и токенизатора для русского языка
model_checkpoint_rus = 'cointegrated/rubert-tiny-toxicity'
tokenizer_rus = AutoTokenizer.from_pretrained(model_checkpoint_rus)
model_rus = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_rus)
if torch.cuda.is_available():
    model_rus.cuda()

# Загрузка модели для английского языка
classifier_eng = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


def text2toxicity(text, aggregate=True):
    with torch.no_grad():
        inputs = tokenizer_rus(text, return_tensors='pt', truncation=True, padding=True).to(model_rus.device)
        proba = torch.sigmoid(model_rus(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba


def get_video_metadata(video_url):
    print("Функция запущена")
    start_time = time.time()  # Начало отсчета времени

    response = requests.get(video_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Извлечение метаданных
    title_tag = soup.find('meta', property='og:title')
    description_tag = soup.find('meta', property='og:description')
    age_restriction_tag = soup.find('meta', property='age-restriction')

    title = title_tag['content'] if title_tag else 'Название не найдено'
    description = description_tag['content'] if description_tag else 'Описание не найдено'
    age_restriction = age_restriction_tag['content'] if age_restriction_tag else '0'  # Предполагаем, что нет ограничений

    # Определение языка текста
    language = detect(description) if description != 'Описание не найдено' else 'unknown'
    print(f"Detected language: {language}")

    if language == 'ru':
        # Анализ метаданных с использованием русской модели
        toxicity_score = text2toxicity(description, True)
        is_safe_for_kids = toxicity_score < 0.5
        toxicity_percentage = toxicity_score * 100
        print(f"Токсичность текста: {toxicity_percentage:.2f}%")
    else:
        # Анализ метаданных с использованием английской модели
        if description != 'Описание не найдено':
            prediction = classifier_eng(description, return_all_scores=True)
            print(f"Prediction: {prediction}")  # Вывод предсказания для отладки
            # Определяем безопасность для детей на основе позитивного/негативного анализа
            is_safe_for_kids = prediction[0][0]['label'] == 'LABEL_1'
        else:
            is_safe_for_kids = int(age_restriction) < 13

    end_time = time.time()  # Конец отсчета времени
    elapsed_time = end_time - start_time  # Вычисление затраченного времени

    return {
        'title': title,
        'description': description,
        'is_safe_for_kids': is_safe_for_kids,
        'elapsed_time': elapsed_time
    }


# Пример использования функции для русского видео
print('Запуск кода')
video_url_rus = 'https://www.youtube.com/watch?v=wl7DE3mNsz0'
metadata_rus = get_video_metadata(video_url_rus)
print(metadata_rus)

# Пример использования функции для английского видео
print('Запуск кода')
video_url_eng = 'https://www.youtube.com/watch?v=LwXrBe4QbMU'
metadata_eng = get_video_metadata(video_url_eng)
print(metadata_eng)


# rus
# 'https://www.youtube.com/watch?v=wl7DE3mNsz0'

# eng
# 'https://www.youtube.com/watch?v=LwXrBe4QbMU'
