LSTM_FEATURE_COLS = [
    # 'return_3',
    'return_20',

    # ===== 추세 변화 =====
    'momentum_3',
    'momentum_20',          
    'momentum_60',          
    'momentum_accel_3',
    'momentum_accel_20',

    # ===== 변동성 흐름 =====
    'atr_change',          
    'volatility_regime_20', 
    'volatility_regime_60', 

    # ===== 거래량 흐름 =====
    # 'volume_ratio',
    'volume_change',
    'volume_zscore_20',    
    'volume_zscore_60',     

    # ===== 캔들 흐름 =====
    'candle_body',
    'high_low_spread',

    # ===== 시장 동조 =====
    'relative_strength',
    'high_breakout_20',
    'high_breakout_60',
    
]
GBM_FEATURE_COLS = [
    # 모멘텀
    'change_rate',
    # 'return_1',
    'return_5',

    'disparity_20',
    # 시장 상대 강도
    'alpha',
    'alpha_5',
    'alpha_20',
    'alpha_divergence',
    # 이동평균
    'ma_ratio',
    # 'price_ma20',
     # RSI / 변동성
    'rsi',
    'volatility_5',
    #볼린저
    'bb_percent',
    #심리도
    # 'psychological',
    #macd
    'macd_hist',
    # 거래량
    'volume_ratio',
    #최고가 대비 하락률
    'drawdown_20',
    # 시장
    # 'nasdaq_change_rate',        
    # 5일간의 고가 - 저가 평균 (종목의 활동성)
    'tr_5'
    
]
