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

    # ===== 시장 동조 =====
    # 'relative_strength',
    # 'high_breakout_20',
    'high_breakout_60',
    # ===== 시장 지수 =====
    # 'vix_vs_stock_vol',
    # 시장 센티멘트 흐름
    'nasdaq_change_rate'
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

TICKERS = [
    # 반도체
    "NVDA", "AMD", "AVGO", "MU", "QCOM", "TSM",

    # 빅테크
    "MSFT", "AMZN", "META", "GOOGL", "AAPL",
    "TSLA", "NFLX", "ORCL", "CRM", "ADBE",

    # AI 인프라
    "PLTR", 
    "NOW",
    # "CRWV",
    "APP", "ANET",

    # 클라우드/사이버
    "CRWD", "PANW", "ZS", "NET", "DDOG", "SNOW",

    # 핀테크/소비
    "COIN", "HOOD", "SOFI", "SHOP", "MELI", "UBER", "ABNB",

    # 우주/방산
    "RKLB",   # 로켓랩
    "ASTS",   # AST SpaceMobile
    "LMT",    # 록히드마틴

    # 에너지/인프라
    "GEV",    # GE Vernova
    "VRT",    # Vertiv
    "NEE",    # NextEra Energy

    # 퀀텀
    "IONQ",
    "OKLO",

    # 기타 성장
    "DASH", 
    "SPOT", 
    "TTD",

    "SMCI",   # Super Micro Computer — AI 서버
    "ARM",    # ARM Holdings — AI 칩 설계
    "WDC",    # Western Digital — SanDisk 모회사 (스토리지)
    "STX",    # Seagate — 데이터센터 스토리지
    "ASML",   # ASML — 반도체 장비 (AI 핵심 공급망)

    # 원자력/에너지 (AI 데이터센터 전력 수요 수혜)
    "VST",    # Vistra — 원자력+전력
    "CEG",    # Constellation Energy — 원자력
    "FSLR",   # First Solar — 태양광

    # 핀테크 추가
    "PYPL",   # PayPal
    "AFRM",   # Affirm — BNPL
    "NU",     # Nubank — 라틴 핀테크

    # 헬스/바이오 (요즘 핫)
    "HIMS",   # Hims & Hers — GLP-1 비만치료
    "ISRG",   # Intuitive Surgical — 수술 로봇

    # 소비자/소셜
    "PINS",   # Pinterest
    "LYFT",   # Lyft
    "RBLX",   # Roblox

    # SaaS
    "INTU",   # Intuit
    "WDAY",   # Workday
    "TEAM",   # Atlassian
]
