# Разработка информационной системы ИИ на базе современных LLM-моделей
### Решение кейса хакатона DubnaTECH (29 марта - 3 апреля 2024) от комании Нордавинд-Дубна

Целью хакатона была разработка чата с большой языковой моделью. 

В качестве базовой модели была выбрана [Mistral-7B-Instruct-v0.2](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ) версии GPTQ, которая потребялет относительно мало памяти и подходит для запуска на одном графическом ускорителе. 

При своем небольшом размере в 7М (миллиардов) параметров, качество ответов базовой модели сравнимо с моделями ChatGPT ранних версий (175М параметров). Недостатком же является то, что модель плохо приспособлена для работы с русским языком. В рамках хакатона качество ответов на русском языке было **улучшено дообучением** методом [QLoRA](https://github.com/artidoro/qlora) на отфильтрованном **датасете сервиса Yandex Q**. 

Дообучение делалось в среде Google Colab на графическом процессоре NVIDIA A100 в течении 24 минут (60 эпох). График Training и Validation Loss можно будет увидеть ниже. Дообученную LoRA-модель я выложил на Hugging Face. 

Поверх языковой модели реализован **поиск по файлам** с помощью метода **RAG (Retrieval Augmented Generation)**: все файлы в директории documents (точно работает с pdf и txt) индексируются, разбиваются по частям и переводятся в векторное представлением с помощью эмбеддинг-модели.
Запросы пользователей переводятся в векторное представление и сравниваются с индексированными файлами, и, в случае близкого рассояния векторов, выбираются k ближайших и подаются на вход модели как результаты поиска. Теоретически это позволяет управлять поиском используя LLM.
Однако сейчас в репозиторий загружены только два файла - мое резюме (чтобы протестировать PDF) и текст романа Мастер и Маргарита, поэтому не удивляйтесь, если вы получите резльутаты связанные с этими файлами. Текст романа очень большой, поэтому, с его текста с большей вероятностью будут выводитьяс при поиске.

Интерфейсом взаимодействия с ботом выступает Telegram-бот.

### Основые ссылки:
- Модель: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ (ветка gptq-4bit-32g-actorder_True)
- Полученная fine-tuned (дообучения) LoRA-модель : https://huggingface.co/ggeorge/qlora-mistral-hackatone-yandexq
- Датасет Yandex Q: https://huggingface.co/datasets/its5Q/yandex-q
- Ссылка на Google Colab обучения модели и подготовки датасета: https://colab.research.google.com/drive/1YpIl1Qbe2pfWChLST7hbJ9FkGdjHF9PR?usp=sharing - ДЛЯ ЗАПУСКА ИСПОЛЬЗУЙТЕ ДРУГОЙ COLLAB, ССЫЛКА БУДЕТ НИЖЕ
- Эмбеддинг-модель, используемая для индексирования файлов для русских и английских слов: https://huggingface.co/cointegrated/LaBSE-en-ru

Далее представленны способы запуска моделей: Google Colab и локальный. 
## Google Colab

Ссылка на Colab notebook для ЗАПУСКА модели: https://colab.research.google.com/drive/1lHbdhq8rJqYoZpMdhFWMj1G1zMlW2qkV?usp=sharing

Как уже было сказано, модель обучалась и тестировалась в среде Google Collab.
Это рекомендуемый способ запуска модели. Бесплатные ресурсы Google Colab позволяют запустить телеграм бота, через которого будет происходить общение.
В среднем модель потребляет около 
- **6.5 GB GPU RAM**
- **5.5 GB System RAM**.

Прежде чем приступить к запуску модели, вам нужно получить Telegram-токен [у BotFather](https://t.me/BotFather) - бота, который отвечает за работу с другими ботами в Телеграме.    

Остальные инструкции есть в самом ноутбуке Google Colab.

Если вы много используете GPU (обычно это NVIDIA T4) в Google Colab, то вам могут на время ограничить доступ к вычислительным ресурсам. Поэтому, если у вас проблема с получением доступа к GPU, попробуйте запустить модель другом аккануте.

Если у вас возникают какие-либо вопросы или проблемы -- не стесняйтесь писать мне в телеграм: [@examoore](https://t.me/examoore)

## Запуск локально

Вам поднадобится:
- интерпретератор Python 3.10 или больше
- операционная система Linux
- графический процессор  с поддержкой CUDA 12.2 (другие версии тоже могут работать, но не тестировались) с объемом памяти >8 ГБ

1. Клонируйте этот репозиторий с помощью команды `git clone https://github.com/ggeorg0/rag-finetuned-mistral7b.git`
2. Создайте вирутальное окружение, чтобы не возникало проблем с зависимостями с другими проектами и установите зависимости
```bash
python3 -m venv ./.venv && source ./.venv/bin/activate
pip3 install -r requirements.txt
```
3. Если вы хотите добавить в систему поиска свои файлы, загрузите их в директорию `documents`.
4. Сохраните токен для телеграм-бота в файл `telegram_token.txt`  Такой способ выбран для упрощения процедуры запуска модели. Вы можете использовать следующую команду, для записи токена в файл. 
```bash
echo "YOUR_TELEGRAM_TOKEN_HERE">telegram_token.txt
```
5. Запустите модель с ботом. Первый запуск может занять какое-то время, так как нужно будет скачать модель. 
```bash
python3 bot.py
```
6. Готово! Вы великолепны. Чтобы выйти из виртуального окружения используйте команду `deactivate` 

## Подготовка датасета и обучение.
[Yandex Q](https://yandex.ru/q/) - это место, где пользователи могут задавать вопросы и отвечать на них. 

Исходный датасет был загружен с [Hugging Face](https://huggingface.co/datasets/its5Q/yandex-q)

Размер исходных данных: **286 MB** -- **около 664 тысячи строчек** с различными вопросами и ответами.


Данные был отфильтрованы по размеру ответа (от 400 до 1600 симв), чтобы не отправлять слишком много слов в языковую модель и избежать отдносложных ответов. Были оставльны только те ответы, у которых есть описание.

```python
low_threshould = 400
high_threshould = 1600

clean_dataset = dataset.filter(lambda ds: low_threshould <= len(ds['answer']) <= high_threshould
                                          and len(ds['description']) > 100)
```

Далее были удалены ответы содержающие обсценную лексику и другие ключевые слова, которые на мой взгляд снижали качество общей выборки. Также удалены предложения с приветсвиями вроде "Здравсвуйте, Семен!"

Таким образом объем данных сократился до **26769 вопросов и ответов**, на которых обучалась модель.

Обучение на графическом процессоре NVIDIA A100 в течении 24 минут (60 эпох)
```yaml
learning_rate: 5e-05
train_batch_size: 8
eval_batch_size: 8
```
