import torch
import numpy as np
import matplotlib.pyplot as plt



def plot_dataset_sample(sample):
    """
    Отрисовывает один sample из полученного датасета.
    На первом графике отрисовываются цены.
    На втором графике отрисовывается объём.
    
    Args:
        sample (dict): Словарь, содержащий ключи 'history', 'target' и 'ticker'.
                       'history' - тензор формы [1, history_len, 5] (Open, High, Low, Close, Volume).
                       'target' - тензор формы [1, target_len, 1] (Close).
                       'ticker' - строка с названием тикера.
    """
    # Извлекаем данные и убираем размерность батча
    history = sample['history'].squeeze(0).numpy()  # [history_len, 5]
    target = sample['target'].squeeze(0).numpy()    # [target_len, 1]
    ticker = sample['ticker']

    # Разделяем исторические данные
    history_len = history.shape[0]
    history_prices = history[:, :4]  # Open, High, Low, Close
    history_volume = history[:, 4]   # Volume

    # Извлекаем цены закрытия из таргета
    target_len = target.shape[0]
    target_close = target[:, 0]      # Close prices for target

    # Создаем ось X для графиков
    history_x = np.arange(history_len)
    target_x = np.arange(history_len, history_len + target_len)

    # --- Первый график: Цены ---
    plt.figure(figsize=(15, 8))

    # Исторические цены
    plt.plot(history_x, history_prices[:, 0], color='gray', alpha=0.7, label='Open (History)') # Open
    plt.plot(history_x, history_prices[:, 1], color='green', alpha=0.7, label='High (History)') # High
    plt.plot(history_x, history_prices[:, 2], color='red', alpha=0.7, label='Low (History)') # Low
    plt.plot(history_x, history_prices[:, 3], color='black', alpha=0.8, label='Close (History)') # Close

    # Разделительная линия
    plt.axvline(x=history_len - 1, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='History/Target Boundary')

    # Цены закрытия таргета
    plt.plot(target_x, target_close, color='black', linewidth=2, label='Close (Target)')

    # Координаты для линии: последняя точка Close из истории -> первая точка Close из таргета
    connect_x = [history_len - 1, history_len]
    connect_y = [history_prices[-1, 3], target_close[0]]
    plt.plot(connect_x, connect_y, color='orange', linewidth=1.5, linestyle='-', marker='o', markersize=4,
             label='Connection (Last Hist Close -> First Target Close)')

    # Настройки первого графика
    plt.title(f'Price History and Target for {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.5)
    # Устанавливаем пределы оси X для первого графика
    plt.xlim(0, history_len + target_len - 1)

    plt.tight_layout()
    plt.show()

    # --- Второй график: Объёмы ---
    plt.figure(figsize=(15, 4))

    # Исторические объёмы
    plt.bar(history_x, history_volume, width=1.0, color='lightblue', alpha=0.7, label='Volume (History)')
    
    # Настройки второго графика
    plt.title(f'Trading Volume History for {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Volume')
    plt.legend()
    plt.grid(True, alpha=0.5, axis='y')
    # Устанавливаем пределы оси X для второго графика (только история)
    plt.xlim(0, history_len + target_len - 1)

    plt.tight_layout()
    plt.show()



def plot_denormalized_sample(sample, score):
    """
    Отрисовывает денормализованные исторические данные и таргет.
    
    Args:
        sample (dict): Словарь с ключами 'history', 'target', 'ticker', 'stats'.
                       'history' и 'target' должны быть numpy массивами или torch тензорами.
                       'stats' должен содержать статистики для денормализации.
    """
    # Извлекаем данные
    history = sample['history']
    target = sample['target']
    ticker = sample['ticker']
    stats = sample['stats']
    
    # Если данные в формате torch.Tensor, конвертируем в numpy
    if hasattr(history, 'numpy'):
        history = history.squeeze(0).numpy()
    if hasattr(target, 'numpy'):
        target = target.squeeze(0).numpy()
    if hasattr(score, 'numpy'):
        score = score.squeeze(0).detach().cpu().numpy()
    
    # Денормализация
    denorm_history, denorm_target = denormalize_sample(history, target, stats)
    
    # Создаем временную шкалу
    history_len = denorm_history.shape[0]
    target_len = denorm_target.shape[0]
    history_time = np.arange(history_len)
    target_time = np.arange(history_len, history_len + target_len)
    
    # Создаем график
    plt.figure(figsize=(12, 6))
    
    # Отрисовка исторических данных (цены закрытия)
    plt.plot(history_time, denorm_history[:, 3], label='Historical Close Price', color='black', linewidth=1)
    plt.plot(history_time, denorm_history[:, 0], label='Historical Open Price', color='gray', linewidth=1)
    plt.plot(history_time, denorm_history[:, 1], label='Historical High Price', color='darkgreen', linewidth=1)
    plt.plot(history_time, denorm_history[:, 2], label='Historical Low Price', color='darkred', linewidth=1)

    # Разделительная линия
    plt.axvline(x=history_len - 1, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='History/Target Boundary')
    
    # Отрисовка таргета (цены закрытия) и его min/max значений
    plt.plot(target_time, denorm_target[:, 0], label='Target Close Price', color='red', linewidth=1)
    
    # Настройка графика
    plt.title(f'[{ticker}] - Denormalized Price Data | Probability: {(score * 100)}%')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Показываем график
    plt.tight_layout()
    plt.show()



def denormalize_sample(history, target, stats):
    """
    Денормализует исторические данные и таргет, используя статистики из нового нормализатора.
    
    Args:
        history (np.ndarray): Исторические данные формы [history_len, 5].
        target (np.ndarray): Таргет данных формы [target_len, 1].
        stats (dict): Статистики нормализации.
        
    Returns:
        tuple: (denorm_history, denorm_target)
    """
    # Копируем данные, чтобы не изменять оригинальные
    denorm_history = np.copy(history)
    denorm_target = np.copy(target)
    
    # Индексы: 0=Open, 1=High, 2=Low, 3=Close, 4=Volume
    price_indices = [0, 1, 2, 3]  # OHLC
    
    # Денормализация цен (O, H, L, C) через лог-доходности
    for i in price_indices:
        # Получаем статистики для денормализации
        first_log_price = stats.get(f'first_log_price_{i}', 0.0)  # Это ln(P_0)
        log_return_mean = stats.get(f'log_return_mean_{i}', 0.0)
        log_return_std = stats.get(f'log_return_std_{i}', 1.0)
        
        # Денормализуем лог-доходности: r_denorm = r_norm * std + mean
        log_returns = denorm_history[:, i]
        if i == 0:  # Для первой точки это ln(P_0), остальные - доходности
            denorm_log_returns = np.copy(log_returns)
            denorm_log_returns[1:] = log_returns[1:] * log_return_std + log_return_mean
        else:
            denorm_log_returns = log_returns * log_return_std + log_return_mean
            # Первая точка - это ln(P_0)
            denorm_log_returns[0] = first_log_price
            
        # Восстанавливаем цены из лог-доходностей
        prices = np.zeros_like(denorm_log_returns)
        prices[0] = np.exp(first_log_price)  # P_0 = exp(ln(P_0))
        
        # Последовательное восстановление цен
        for t in range(1, len(denorm_log_returns)):
            # P_t = P_{t-1} * exp(r_t)
            prices[t] = prices[t-1] * np.exp(denorm_log_returns[t])
            
        denorm_history[:, i] = prices
    
    # Денормализация таргета (цены закрытия)
    # Используем те же статистики, что и для исторических данных цены закрытия
    first_log_price = stats.get('first_log_price_3', 0.0)
    log_return_mean = stats.get('log_return_mean_3', 0.0)
    log_return_std = stats.get('log_return_std_3', 1.0)
    
    # Для таргета денормализуем лог-доходности
    target_log_returns = denorm_target[:, 0]
    denorm_target_log_returns = target_log_returns * log_return_std + log_return_mean
    
    # Восстанавливаем цены таргета
    target_prices = np.zeros_like(denorm_target_log_returns)
    
    # Начинаем с последней цены в денормализованной истории
    last_history_price = denorm_history[-1, 3]  # Последняя цена закрытия в истории
    
    # Первую цену таргета вычисляем из последней цены истории и первой доходности таргета
    target_prices[0] = last_history_price * np.exp(denorm_target_log_returns[0])
    
    # Последовательное восстановление остальных цен таргета
    for t in range(1, len(denorm_target_log_returns)):
        target_prices[t] = target_prices[t-1] * np.exp(denorm_target_log_returns[t])
        
    denorm_target[:, 0] = target_prices
    
    # Денормализация объема
    volume_mean = stats.get('volume_mean', 0.0)
    volume_std = stats.get('volume_std', 1.0)
    
    # Обратное z-score преобразование
    denorm_history[:, 4] = denorm_history[:, 4] * volume_std + volume_mean
    
    return denorm_history, denorm_target
