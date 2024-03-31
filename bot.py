from __future__ import annotations

import os
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)


from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes
from telegram.ext import filters

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
MODEL_REVISION = "gptq-4bit-32g-actorder_True"
EMBEDDING_MODEL = "cointegrated/LaBSE-en-ru"

PRETRAINED_LORA = "ggeorge/qlora-mistral-hackatone-yandexq"

TG_TOKEN_PATH = "telegram_token.txt"

STOPWORDS_DIRECTORY = "stopwords"

# device: 'auto' or 'gpu'
DEVICE = 'auto'

# your documents directory
STORAGE = "documents"

SYSTEM_PROMPT = "Вы - русскоязычный ИИ ассистент, который помогает пользователям находить файлы, \
отвечать на их вопросы и поддерживать диалог. \
Вы имеете доступ к базе знаний, которая автоматически покажет результаты поиска, \
если найдется какая-либо информация. Помните, что не все результаты поиска могут быть релевантны. \
Адаптируйтесь к запросам пользователей и предоставляйте понятные и полезные ответы на вопросы, \
поддерживая диалог на русском языке. \
\n\
\n\
"

INTRUCT_TEMPLATE = "[INST]{sys_inst}{context}\n\nСообщение пользователя:\n{message}[/INST]"

TG_GREET_MESSAGE = """Привет! Я большая языковая модель, которая может генерировать текст. 
В мое основе лежит большая языковыя модель Mistral-7B-Instruct-v0.2 (версия GPTQ). Я дообучена на датасете сервиса Yandex Q с использованием QLoRA.
Просто отправь мне свое сообщение, и я отвечу.
"""

DEFAULT_MAX_TOKENS = 250

# type annotations can be inaccurate here
model: PeftModel
model_tokenizer: AutoTokenizer
vector_storage_index: VectorStoreIndex
vector_query_engine: RetrieverQueryEngine
user_dialogs: dict[int, str] = dict()
stopwords: list[str]


def read_telegram_token(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            token = file.read().strip()
    else:
        token = input("File 'telegram_token.txt' not found. Enter token manually: ").strip()
    return token

def load_stopwords(directory):
    stop_words = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            lines = file.readlines()
            stop_words.extend(lines)
    return list(map(str.strip, stop_words))

def load_vector_storage(path_dir, top_k=3):
    global vector_query_engine

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = None
    Settings.chunk_size = 256
    Settings.chunk_overlap = 12

    documents = SimpleDirectoryReader(path_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    vector_query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)
    
def remove_stop_words(query: str) -> str:
    for sw in stopwords:
        query = query.replace(sw, '')
    return query

def knowlage_db_context(query: str) -> str:
    clear_query = remove_stop_words(query)
    search_response = vector_query_engine.query(clear_query)
    if not search_response.source_nodes:
        return 'В базе знаний ничего не найдено'
    context = ['Найдено в базе знаний:\n']
    for node in search_response.source_nodes:
        context.append( f'\tимя файла: {node.metadata["file_name"]}\n' )
        context.append( f'\tдата создания: {node.metadata["creation_date"]}\n' )
        context.append( f'\tтекст: {node.text}' + "\n" )

    merged_context = '\n'.join(context)
    logging.info(f"Model context: {merged_context}")
    return merged_context

    
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        trust_remote_code=False,
        revision=MODEL_REVISION)

    config = PeftConfig.from_pretrained(PRETRAINED_LORA)
    model = PeftModel.from_pretrained(model, PRETRAINED_LORA)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer

def generate_inital_prompt(user_query):
    return INTRUCT_TEMPLATE.format(
        sys_inst=SYSTEM_PROMPT,
        context=knowlage_db_context(user_query),
        message=user_query)

def continue_dialog(history, user_query):
    return history + '\n' + INTRUCT_TEMPLATE.format(
        sys_inst='\n',
        context=knowlage_db_context(user_query,
        message=user_query)
    )

def query_model(prompt) -> str:
    global model_tokenizer
    inputs = model_tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                             max_new_tokens=DEFAULT_MAX_TOKENS)
    return model_tokenizer.batch_decode(outputs)[0]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(TG_GREET_MESSAGE)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(TG_GREET_MESSAGE)

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global user_dialogs
    chat_id = update.effective_chat.id
    user_dialogs.pop(chat_id)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global user_dialogs
    chat_id = update.effective_chat.id
    new_text = update.message.text
    if chat_id in user_dialogs:
        prompt = continue_dialog(user_dialogs[chat_id], new_text)
    else:
        prompt = generate_inital_prompt(new_text)
    try:
        model_output = query_model(prompt)
        logging.info(f'Model output: {model_output}')
        user_dialogs[chat_id] = model_output
        last_inst = model_output.rfind('[/INST]')
        await context.bot.send_message(chat_id, model_output[last_inst+7:])
    except Exception as e:
        await context.bot.send_message(chat_id, '[!] Произошла ошибка, смотрите логи!')
        logging.exception(f"cannot generate output, reason:\n")


def main():
    global vector_storage_index, model, model_tokenizer, stopwords
    token = read_telegram_token(TG_TOKEN_PATH)
    if not token:
        print('Telegram token is empty')
        sys.exit(-1)

    stopwords = load_stopwords(STOPWORDS_DIRECTORY)

    logging.info(f'indexing documents in the direcotry: {STORAGE}')
    vector_storage_index = load_vector_storage(STORAGE, top_k=3)

    logging.info(f'loading model: {MODEL_NAME} {MODEL_REVISION}')
    logging.info(f'qlora: {PRETRAINED_LORA}')
    model, model_tokenizer = load_model()

    logging.info(f'building telegram bot')
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.getLogger('httpx').setLevel(logging.WARNING)
    app.run_polling()

if __name__ == "__main__":
    main()
