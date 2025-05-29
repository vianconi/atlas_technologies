import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext,
    load_index_from_storage, Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

DATA_DIR = './data'
PERSIST_DIR = './storage'

def setup_models():
    api_key = os.getenv('OPENAI_API_KEY')
    try:
        Settings.llm = OpenAI(model='gpt-3.5-turbo', api_key=api_key)
        Settings.embed_model = OpenAIEmbedding(api_key=api_key)
        print('Modelos OpenAI configurados con éxito')
    except Exception as e:
        print(f"Error configurando modelos: {e}")
        raise

def get_index(data_path, persist_path):
    if not os.path.exists(persist_path) or not os.listdir(persist_path):
        try:
            documents = SimpleDirectoryReader(data_path).load_data()
            index = VectorStoreIndex.from_documents(
                documents, show_progress=True
            )
            index.storage_context.persist(persist_dir=persist_path)
        except Exception as e:
            print(f"Error creando índice: {e}")
            raise
    else:
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=persist_path
            )
            index = load_index_from_storage(storage_context)
        except Exception as e:
            print(f"Error cargando índice: {e}")
            raise
            
    return index

def run_chat(index):
    custom_system_prompt = (
        'Comportate como un profesor de derecho, y responde solamente con base '
        'al contexto proveído por los documentos, no uses tu conocimiento '
        'previamente adquirido, si la respuesta no se encuentra en los '
        'documentos, simplemente responde que no sabes.'
    )

    chat_engine = index.as_chat_engine(
        chat_mode='context',
        system_prompt=custom_system_prompt,
    )

    while True:
        try:
            query = input('\nUsted: ')
            response = chat_engine.chat(query)
            print(f'\nBot: {response.response}')
        except Exception as e:
            print(f"Error en la conversación: {e}")
            raise

if __name__ == "__main__":
    print('Iniciando la aplicación RAG de terminal...')

    setup_models()

    rag_index = get_index(DATA_DIR, PERSIST_DIR)

    run_chat(rag_index)

    print('Aplicación cerrada.')
    