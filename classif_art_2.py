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

# Загрузка данных
# Предполагается, что у вас есть CSV файл с колонками "title", "description" и "genre"
df = pd.read_csv('dataset_1.csv', encoding='cp1251', header=0, sep=';')
print(df.keys().tolist())
print(df)
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

# Оценка модели
#y_pred = pipeline.predict(X_test)
#print(classification_report(y_test, y_pred))

# Использование модели для предсказания жанра нового видео
def predict_genre(title, description):
    text = preprocess_text(title + ' ' + description)
    return pipeline.predict([text])[0]

# Пример использования
new_title = "Кровавые распри"
new_description = "Документальное видео о насилии в одном невымышленном городке"
print(predict_genre(new_title, new_description))
df_2 = pd.read_csv('dataset_ex.csv', encoding='cp1251', sep=';')
for index, row in df_2.iterrows():
    new_title = row['title']
    new_description = row['text']
    predicted_genre = predict_genre(new_title, new_description)
    print(f"Название: {new_title},  Предсказанный жанр: {predicted_genre}")