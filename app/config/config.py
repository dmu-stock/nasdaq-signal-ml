LSTM_FEATURE_COLS = [
    
    # 'return_3',
    # 'return_20',

    # ===== 추세 변화 =====
    # 'momentum_3',
    # 'momentum_20',          
    # 'momentum_60',          
    # 'momentum_accel_3',
    # 'momentum_accel_20',

    # ===== 변동성 흐름 =====
    # 'atr_change',          
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

    # 레짐 인지 (핵심)
    # 'momentum_regime_adj',
    # 'mom_factor_20',

]



GBM_FEATURE_COLS = [
    # 모멘텀
    # 'change_rate',
    # 'return_1',
    'return_5',
    'nasdaq_change_rate',
    # 이격도
    'disparity_20',
    # 시장 상대 강도
    # 'alpha',
    'alpha_5',
    # 'alpha_20',
    # 'alpha_divergence',
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
    # 5일간의 고가 - 저가 평균 (종목의 활동성)
    'tr_5',
    'tr_20',
    'tr_60',
    'pullback_zscore',
    'price_position_52w',
    # 'disparity_zscore'
    # 시장 센티멘트
    # 'obv_slope_5',   # ← 추가
    # 'mfi',  
]

TICKERS = [   # 나스닥 41 (활성)
    # AI 반도체 (핵심)
    "NVDA", "AMD", "AVGO", "MU", "ARM", "SMCI",

    # 빅테크
    "MSFT", "META", "GOOGL", "AMZN", "AAPL", "TSLA", "NFLX",

    # AI 인프라/소프트웨어
    "PLTR", "NOW", "APP", "ANET", "CRM",

    # 클라우드/사이버
    "CRWD", "PANW", "DDOG", "NET",

    # 전력 인프라 (AI 데이터센터 수혜)
    "VST",    # Vistra — 원자력
    "CEG",    # Constellation Energy — 원자력
    "GEV",    # GE Vernova — 전력
    "VRT",    # Vertiv — 데이터센터 냉각

    # 핀테크/크립토
    "COIN", "HOOD", "NU", "AFRM",

    # 헬스테크
    "HIMS",   # GLP-1 수혜

    # 우주/방산
    "RKLB", "ASTS",

    # 퀀텀/소형 원자력
    "IONQ", "OKLO",

    # 소비/성장
    "UBER", "SPOT", "DASH", "RBLX",

    # 스토리지 (AI 데이터)
    "WDC", "STX",
]
TICKERS_SP = [   # S&P 39 (백업)
    # 금융 (당신엔 없음)
    "JPM", "BAC", "WFC", "GS", "MS",

    # 헬스케어 (당신엔 없음)
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE",

    # 필수소비재
    "PG", "KO", "PEP", "WMT", "COST",

    # 산업재
    "CAT", "BA", "HON", "GE", "UPS",

    # 에너지
    "XOM", "CVX", "COP",

    # 소재
    "LIN", "SHW", "FCX",

    # 통신 (전통)
    "VZ", "T", "CMCSA",

    # 유틸리티
    "DUK", "SO", "NEE",

    # 부동산
    "AMT", "PLD",

    # 임의소비재 (전통)
    "MCD", "NKE", "HD", "SBUX",
]
