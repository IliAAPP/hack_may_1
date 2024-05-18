import requests
from bs4 import BeautifulSoup
import time
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


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
    age_restriction = age_restriction_tag[
        'content'] if age_restriction_tag else '0'  # Предполагаем, что нет ограничений

    # Анализ метаданных с использованием модели
    if description != 'Описание не найдено':
        prediction = classifier(description, return_all_scores=True)
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


# Пример использования функции
print('Запуск кода')
video_url = 'https://www.youtube.com/watch?v=LwXrBe4QbMU'
metadata = get_video_metadata(video_url)
print(metadata)

# список проверенных видео, моя оценка (негатив - 1), процент уверенности в негативе у модели
# https://www.youtube.com/watch?v=ac4E_UsmB1g - 1 - 0.9969853758811951
# https://www.youtube.com/watch?v=cIm-1d5Q6sQ - 0 - 0.012497405521571636
# https://www.youtube.com/watch?v=X6mll8rgDSI - 1 - 0.9994686245918274
# https://www.youtube.com/watch?v=LwXrBe4QbMU - 1 - 0.9984116554260254
# https://www.youtube.com/watch?v=f_WCg5WpFC8 - 0 - 0.0018170495750382543
# https://www.youtube.com/watch?v=jfKfPfyJRdk - 0 - 0.9777838587760925
# https://www.youtube.com/watch?v=1xiA4JBXJyw - 1 - 0.9996756315231323
# https://www.youtube.com/watch?v=Q10tVT5aF2Y - 1 - 0.9847354888916016
# https://www.youtube.com/watch?v=_XI445p288s - 0 - 0.9985413551330566
# https://www.youtube.com/watch?v=tzgoYDHydBs - 1 - 0.47053083777427673
# https://www.youtube.com/watch?v=G6k7dChBaJ8 - 0 - 0.00037559805787168443
# https://www.youtube.com/watch?v=TX2X9XRLOSw - 0 - 0.25187909603118896
# https://www.youtube.com/watch?v=ZyhrYis509A - 0 - 0.5457535982131958
