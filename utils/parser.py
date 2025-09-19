# utils/parser.py
import os
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm



def split_into_chunks(data, chunk_len, step=1):
    """
    Разбивает данные на чанки заданной длины.
    
    Args:
        data: numpy массив с данными
        chunk_len: длина одного чанка
        step: шаг между чанками
        
    Returns:
        list: список чанков
    """
    chunks = []
    for i in range(0, len(data) - chunk_len + 1, step):
        chunk = data[i:i + chunk_len]
        chunks.append(chunk)
    return chunks



def parse_single_ticker(
    ticker, 
    path_to_save='data/', 
    timeframe='1d',
    start_date='2020-01-01',
    target_len=32, 
    history_len=256, 
    split_coef=0.1,
):
    """
    Парсит данные для одного тикера и сохраняет их в виде чанков.
    
    Args:
        ticker (str): Тикер акции
        path_to_save (str): Путь для сохранения данных
        timeframe (str): Таймфрейм данных
        start_date (str): Начальная дата
        target_len (int): Длина таргета
        history_len (int): Длина истории
        split_coef (float): Коэффициент разбиения на train/val
    """
    # Получаем историю цен
    stock = yf.Ticker(ticker)
    history = stock.history(
        interval=timeframe, 
        start=start_date, 
        actions=False, 
        auto_adjust=True, 
        prepost=True
    )

    # Преобразуем в numpy массив (OHLCV)
    data_values = history.values

    # Разбиваем историю на чанки длины history_len + target_len
    chunk_size = history_len + target_len
    chunks = split_into_chunks(data_values, chunk_size)

    # Разделяем чанки на тренировочный и валидационный наборы (хронологически)
    split_index = int(len(chunks) * (1 - split_coef))
    train_chunks = chunks[:split_index]
    val_chunks = chunks[split_index:]

    # Создаем директории для сохранения
    train_ticker_path = os.path.join(path_to_save, 'train', ticker)
    val_ticker_path = os.path.join(path_to_save, 'validation', ticker)
    
    os.makedirs(train_ticker_path, exist_ok=True)
    os.makedirs(val_ticker_path, exist_ok=True)

    # Сохраняем тренировочные чанки
    for i, chunk in enumerate(train_chunks):
        pd.DataFrame(chunk).to_csv(
            os.path.join(train_ticker_path, f'chunk_{i}.csv'), 
            index=False, header=False
        )
    
    # Сохраняем валидационные чанки
    for i, chunk in enumerate(val_chunks):
        pd.DataFrame(chunk).to_csv(
            os.path.join(val_ticker_path, f'chunk_{i}.csv'), 
            index=False, header=False
        )



def parse_snp500(
    path_to_save='data/', 
    timeframe='1d',
    start_date='2020-01-01',
    target_len=32, 
    history_len=256, 
    split_coef=0.1,
):
    """
    Парсит данные для всех тикеров S&P 500.
    
    Args:
        path_to_save (str): Путь для сохранения данных
        timeframe (str): Таймфрейм данных
        start_date (str): Начальная дата
        target_len (int): Длина таргета
        history_len (int): Длина истории
        split_coef (float): Коэффициент разбиения на train/val
    """
    # Проверяем, существует ли директория, и создаем её, если нет
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # Если нет таблицы с тикерами всех акций, то её нужно спарсить
    snp500_tickers_path = os.path.join(path_to_save, 'snp500_tickers.csv')
    if not os.path.exists(snp500_tickers_path):
        import requests
        # Используем requests с заголовком User-Agent для обхода блокировки
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status() # Проверка на успешный статус ответа
            # Передаем текст HTML в read_html
            sp500_table = pd.read_html(response.text)[0]
            
            # Убедимся, что директория существует
            os.makedirs(path_to_save, exist_ok=True)
            
            # Сохраняем в csv таблицу матрицу ['Symbol', 'Security']
            sp500_table[['Symbol', 'Security']].to_csv(snp500_tickers_path, index=False, header=False)
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке данных с Wikipedia: {e}")
            raise
        except Exception as e:
            print(f"Ошибка при парсинге таблицы: {e}")
            raise

    # Чтение списка тикеров из файла CSV в датафрейм
    tickers_df = pd.read_csv(snp500_tickers_path, header=None, names=['Symbol', 'Security'])

    # Проходимся по всему списку тикеров
    for index, row in tqdm(tickers_df.iterrows(), total=len(tickers_df)):
        try:
            parse_single_ticker(
                ticker=row['Symbol'],
                path_to_save=path_to_save,
                timeframe=timeframe,
                start_date=start_date,
                target_len=target_len,
                history_len=history_len,
                split_coef=split_coef,
            )
        except Exception as e:
            print(f"Ошибка при парсинге тикера {row['Symbol']}: {e}")
            continue



# Пример использования:
if __name__ == "__main__":
    # Парсинг данных для всех тикеров S&P 500
    parse_snp500(
        path_to_save='data/', 
        timeframe='1d',
        start_date='2020-01-01',
        target_len=32, 
        history_len=256, 
        split_coef=0.1,
    )