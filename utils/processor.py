import numpy as np


class TPEProcessor:
    def __init__(self):
        pass


    def __call__(self, chunk, history_len=256):
        score = self._calculate_score(chunk, history_len)
        normalized_chunk, stats = self._log_return_normalize(chunk, history_len)

        return normalized_chunk, stats, score
    

    def _calculate_score(self, chunk, history_len):
        # Разделяем чанк на историю и таргет
        history_data = chunk[:history_len]
        target_data = chunk[history_len:]

        # Получаем только цены закрытия таргета
        target_close_prices = target_data[:, 3]
        
        # Получаем последнюю цену закрытия в истории
        last_close_price = history_data[-1, 3]

        # Получаем максимум и минимум 
        target_min = np.min(target_close_prices)
        if target_min > last_close_price:
            target_min = last_close_price

        target_max = np.max(target_close_prices)
        if target_max < last_close_price:
            target_max = last_close_price

        # Считаем логарифмы доходности и вероятность роста
        log_return_max = np.log(target_max/last_close_price)
        log_return_min = np.log(target_min/last_close_price)
        PROBABILITY = log_return_max / (log_return_max - log_return_min)

        return np.array([PROBABILITY])
    

    def _log_return_normalize(self, chunk, history_len=256):
        normalized_chunk = np.zeros_like(chunk, dtype=np.float32)
        stats = {}
        
        # Индексы: 0=Open, 1=High, 2=Low, 3=Close, 4=Volume
        price_indices = [0, 1, 2, 3]  # OHLC
        volume_index = 4              # Volume
        
        # Нормализация цен (O, H, L, C)
        for i in price_indices:
            # Вычисляем лог-доходности для всего чанка: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
            prices = chunk[:, i]
            log_prices = np.log(prices + 1e-8)  # Добавляем маленькое число для избежания log(0)
            
            # Для первой точки используем оригинальное значение лог-цены
            log_returns = np.zeros_like(log_prices)
            log_returns[0] = log_prices[0] # Это ln(P_0)
            if len(log_prices) > 1:
                log_returns[1:] = np.diff(log_prices)  # log(P_t) - log(P_{t-1})
                
            # Вычисляем статистики по истории (первые history_len точек)
            history_log_returns = log_returns[:history_len] # [history_len]
            mean_history = np.mean(history_log_returns[1:]) # Исключаем первую точку, так как она это ln(P_0)
            std_history = np.std(history_log_returns[1:])   # Исключаем первую точку
            
            # Избегаем деления на ноль
            if std_history < 1e-8:
                std_history = 1.0
                
            # Нормализуем лог-доходности (всё, включая таргет) по статистикам истории
            # Для первой точки оставляем как есть (это ln(P_0)), для остальных нормализуем
            normalized_log_returns = np.copy(log_returns)
            normalized_log_returns[1:] = (log_returns[1:] - mean_history) / std_history
            
            normalized_chunk[:, i] = normalized_log_returns
            
            # Сохраняем статистики для денормализации
            stats[f'first_log_price_{i}'] = log_prices[0] # ln(P_0)
            stats[f'log_return_mean_{i}'] = mean_history
            stats[f'log_return_std_{i}'] = std_history
            
        # Нормализация объема по z-score по всему чанку
        volumes = chunk[:, volume_index]
        volume_mean = np.mean(volumes)
        volume_std = np.std(volumes)
        
        # Избегаем деления на ноль
        if volume_std < 1e-8:
            volume_std = 1.0
            
        normalized_volumes = (volumes - volume_mean) / volume_std
        normalized_chunk[:, volume_index] = normalized_volumes
        
        # Сохраняем статистики объема для денормализации
        stats['volume_mean'] = volume_mean
        stats['volume_std'] = volume_std
        
        return normalized_chunk, stats
