# utils/crypto_parser.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
import time
from datetime import datetime, timedelta

# Импортируем клиент Binance
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Импортируем вашу функцию разбиения на чанки
from utils.parser import split_into_chunks



def get_all_binance_usdt_symbols():
    """
    Получает список всех торговых пар на Binance, где котировка — USDT.
    
    Returns:
        list: Список символов (например, ['BTCUSDT', 'ETHUSDT', ...])
    """
    try:
        # Создаем клиент без ключей для публичных данных
        client = Client()
        exchange_info = client.get_exchange_info()
        
        # Фильтруем только пары с котировкой USDT
        usdt_symbols = [
            s['symbol'] 
            for s in exchange_info['symbols'] 
            if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
        ]
        
        print(f"Найдено {len(usdt_symbols)} торговых пар против USDT.")
        return usdt_symbols
        
    except Exception as e:
        print(f"Ошибка при получении списка пар с Binance: {e}")
        return []



def parse_single_crypto(
    symbol, 
    path_to_save='data_crypto/', 
    timeframe='1d',
    start_date='2020-01-01',
    target_len=32, 
    history_len=256, 
    split_coef=0.1,
):
    """
    Парсит данные для одной криптовалютной пары с Binance и сохраняет их в виде чанков.
    
    Args:
        symbol (str): Символ пары (например, 'BTCUSDT')
        path_to_save (str): Путь для сохранения данных
        timeframe (str): Таймфрейм данных ('1d', '4h', '1h' и т.д.)
        start_date (str): Начальная дата в формате 'YYYY-MM-DD'
        target_len (int): Длина таргета
        history_len (int): Длина истории
        split_coef (float): Коэффициент разбиения на train/val
    """
    # Создаем клиент Binance
    client = Client()  # Для публичных данных ключи не требуются
    
    try:
        # Конвертируем start_date в timestamp
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = None  # Получаем до последней доступной свечи

        # Определяем лимит для одного запроса (максимум 1000 свечей)
        limit = 1000
        
        # Список для хранения всех свечей
        all_klines = []
        
        # Получаем данные в цикле, чтобы обойти лимит в 1000 свечей
        current_start_ts = start_ts
        while True:
            try:
                klines = client.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    startTime=current_start_ts,
                    endTime=end_ts,
                    limit=limit
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Обновляем start_ts для следующего запроса (время последней свечи + 1 мс)
                last_kline_time = klines[-1][0]
                current_start_ts = last_kline_time + 1
                
                # Задержка для соблюдения лимитов API
                time.sleep(0.1)
                
            except BinanceAPIException as e:
                if e.code == -1121:  # Invalid symbol
                    print(f"Неверный символ: {symbol}")
                    return
                elif e.code == -1003:  # Too many requests
                    print("Слишком много запросов. Ждем 60 секунд...")
                    time.sleep(60)
                    continue
                else:
                    print(f"Ошибка API Binance при парсинге {symbol}: {e}")
                    break
            except Exception as e:
                print(f"Неизвестная ошибка при получении данных для {symbol}: {e}")
                break

        if len(all_klines) < history_len + target_len:
            print(f"Недостаточно данных для {symbol}. Получено {len(all_klines)} свечей, требуется минимум {history_len + target_len}.")
            return

        # Преобразуем свечи в DataFrame
        # Формат свечи: [Open time, Open, High, Low, Close, Volume, Close time, ...]
        df = pd.DataFrame(all_klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        
        # Преобразуем в числовой тип и выбираем нужные столбцы (OHLCV)
        data_values = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float).values

        # Разбиваем историю на чанки длины history_len + target_len
        chunk_size = history_len + target_len
        chunks = split_into_chunks(data_values, chunk_size)

        if len(chunks) == 0:
            print(f"Не удалось создать чанки для {symbol}.")
            return

        # Разделяем чанки на тренировочный и валидационный наборы (хронологически)
        split_index = int(len(chunks) * (1 - split_coef))
        train_chunks = chunks[:split_index]
        val_chunks = chunks[split_index:]

        # Создаем директории для сохранения
        train_ticker_path = os.path.join(path_to_save, 'train', symbol)
        val_ticker_path = os.path.join(path_to_save, 'validation', symbol)
        
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

        print(f"Успешно спарсил и сохранил {len(chunks)} чанков для {symbol}.")

    except Exception as e:
        print(f"Критическая ошибка при парсинге {symbol}: {e}")



def parse_all_binance_usdt(
    path_to_save='data_crypto/', 
    timeframe='1d',
    start_date='2020-01-01',
    target_len=32, 
    history_len=256, 
    split_coef=0.1,
):
    """
    Парсит данные для всех криптовалютных пар против USDT на Binance.
    
    Args:
        path_to_save (str): Путь для сохранения данных
        timeframe (str): Таймфрейм данных
        start_date (str): Начальная дата
        target_len (int): Длина таргета
        history_len (int): Длина истории
        split_coef (float): Коэффициент разбиения на train/val
    """
    # Получаем список всех пар
    symbols = get_all_binance_usdt_symbols()
    
    if not symbols:
        print("Не удалось получить список торговых пар. Завершение.")
        return

    # Проходимся по всему списку символов
    for symbol in tqdm(symbols, desc="Парсинг криптовалют"):
        try:
            parse_single_crypto(
                symbol=symbol,
                path_to_save=path_to_save,
                timeframe=timeframe,
                start_date=start_date,
                target_len=target_len,
                history_len=history_len,
                split_coef=split_coef,
            )
        except Exception as e:
            print(f"Ошибка при парсинге символа {symbol}: {e}")
            continue

    print("Парсинг всех криптовалют завершен.")



# Пример использования:
if __name__ == "__main__":
    # Парсинг данных для всех криптовалютных пар против USDT на Binance
    parse_all_binance_usdt(
        path_to_save='data_crypto/', 
        timeframe='1d',
        start_date='2020-01-01',
        target_len=32, 
        history_len=256, 
        split_coef=0.1,
    )