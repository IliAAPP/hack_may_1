import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import time

# Загрузка модели и токенизатора
model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

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

    end_time = time.time()  # Конец отсчета времени
    elapsed_time = end_time - start_time  # Вычисление затраченного времени

    return {
        'title': title,
        'description': description,
        'age_restriction': age_restriction,
        'elapsed_time': elapsed_time  # Добавление затраченного времени в результат
    }

def text2toxicity(text, aggregate=True):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

# Пример использования функции
print('Запуск кода')
video_url = 'https://www.youtube.com/watch?v=wl7DE3mNsz0'
metadata = get_video_metadata(video_url)
description = metadata['description']
print(f"Description: {description}")

toxicity_score = text2toxicity(description, True)
print(f"Контент небезопасен на {round(toxicity_score * 100, 2)} %")

# https://www.youtube.com/watch?v=s1SO__WJPYk - 0 - 15
# https://www.youtube.com/watch?v=P4UTas8nZrI - 0 - 28
# https://www.youtube.com/watch?v=Gsft0tcbPeY - 0 - 8
# https://www.youtube.com/watch?v=5_HXXqDeuUg - 0 - 11
# https://www.youtube.com/watch?v=_dwGhpeYlPE - 1 - 83
# https://www.youtube.com/watch?v=kjR_VrOMW14 - 1 - 93
# https://www.youtube.com/watch?v=aBm0OMvUPKI - 1 - 65
# https://www.youtube.com/watch?v=_JH3xxPCTr4 - 0 - 35
# https://www.youtube.com/watch?v=wl7DE3mNsz0 - 0 - 3
# https://www.youtube.com/watch?v=Ab3Qz1cA2Do - 0 - 3
# https://www.youtube.com/watch?v=z6w0tWCQVtc - 0 - 2
# https://www.youtube.com/watch?v=NcWk1JJlYko - 0 - 33
# https://www.youtube.com/watch?v=-s9N4mvcSNQ - 0 - 6
# https://www.youtube.com/watch?v=M_dmsMAvmnk - 1 - 98
# https://www.youtube.com/watch?v=v5yqhpA0nNs - 1 - 76
