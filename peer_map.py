"""
Peer stock map — hardcoded competitor sets for top NSE F&O stocks.

Each entry maps a symbol to its 3-5 closest comparable peers. When the user
clicks "Find Better Alternatives" on a trade plan, the API compares the
current stock against these peers head-to-head.

Maintenance: this list should be reviewed quarterly. Mergers and
demergers (like Tata Motors → TMPV+TMCV in 2025) require updates.

For symbols NOT in this map, the API falls back to `ticker.info['sector']`
from Yahoo Finance and finds same-sector stocks from `stock_list.NSE_STOCKS`.
"""

PEER_MAP = {
    # ===== Private Banks =====
    "HDFCBANK":   ["ICICIBANK", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
    "ICICIBANK":  ["HDFCBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK"],
    "KOTAKBANK":  ["HDFCBANK", "ICICIBANK", "AXISBANK", "INDUSINDBK"],
    "AXISBANK":   ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "INDUSINDBK"],
    "INDUSINDBK": ["HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK"],
    "IDFCFIRSTB": ["HDFCBANK", "AXISBANK", "INDUSINDBK", "YESBANK"],
    "YESBANK":    ["IDFCFIRSTB", "INDUSINDBK", "AXISBANK"],

    # ===== Public Sector Banks =====
    "SBIN":       ["BANKBARODA", "PNB", "CANBK"],
    "BANKBARODA": ["SBIN", "PNB", "CANBK"],
    "PNB":        ["SBIN", "BANKBARODA", "CANBK"],
    "CANBK":      ["SBIN", "BANKBARODA", "PNB"],

    # ===== IT Services =====
    "TCS":        ["INFY", "WIPRO", "HCLTECH", "TECHM"],
    "INFY":       ["TCS", "WIPRO", "HCLTECH", "TECHM"],
    "WIPRO":      ["TCS", "INFY", "HCLTECH", "TECHM"],
    "HCLTECH":    ["TCS", "INFY", "WIPRO", "TECHM"],
    "TECHM":      ["TCS", "INFY", "WIPRO", "HCLTECH"],
    "LTIM":       ["TCS", "INFY", "MPHASIS", "PERSISTENT"],
    "MPHASIS":    ["LTIM", "COFORGE", "PERSISTENT", "TECHM"],
    "COFORGE":    ["LTIM", "MPHASIS", "PERSISTENT"],
    "PERSISTENT": ["LTIM", "MPHASIS", "COFORGE"],
    "OFSS":       ["TCS", "INFY", "MPHASIS"],

    # ===== Auto =====
    "MARUTI":     ["TMPV", "M&M", "EICHERMOT", "TVSMOTOR"],
    "TMPV":       ["MARUTI", "M&M", "EICHERMOT", "HEROMOTOCO"],
    "TMCV":       ["ASHOKLEY", "M&M", "EICHERMOT"],
    "M&M":        ["MARUTI", "TMPV", "TVSMOTOR", "EICHERMOT"],
    "EICHERMOT":  ["HEROMOTOCO", "BAJAJ-AUTO", "TVSMOTOR", "M&M"],
    "HEROMOTOCO": ["BAJAJ-AUTO", "TVSMOTOR", "EICHERMOT"],
    "BAJAJ-AUTO": ["HEROMOTOCO", "TVSMOTOR", "EICHERMOT"],
    "TVSMOTOR":   ["BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
    "ASHOKLEY":   ["TMCV", "M&M"],
    "MOTHERSON":  ["BOSCHLTD"],
    "BOSCHLTD":   ["MOTHERSON"],

    # ===== Energy / Oil & Gas =====
    "RELIANCE":   ["ONGC", "BPCL", "IOC", "GAIL"],
    "ONGC":       ["RELIANCE", "BPCL", "IOC", "GAIL"],
    "BPCL":       ["IOC", "ONGC", "GAIL", "RELIANCE"],
    "IOC":        ["BPCL", "ONGC", "GAIL"],
    "GAIL":       ["ONGC", "BPCL", "IOC"],

    # ===== Power =====
    "NTPC":       ["POWERGRID", "TATAPOWER", "ADANIPOWER", "NHPC"],
    "POWERGRID":  ["NTPC", "TATAPOWER", "NHPC"],
    "TATAPOWER":  ["NTPC", "ADANIPOWER", "POWERGRID"],
    "ADANIPOWER": ["NTPC", "TATAPOWER", "POWERGRID"],
    "NHPC":       ["NTPC", "POWERGRID", "SJVN"],
    "ADANIGREEN": ["TATAPOWER", "ADANIPOWER", "SUZLON"],
    "SUZLON":     ["ADANIGREEN", "TATAPOWER"],

    # ===== Metals & Mining =====
    "TATASTEEL":  ["JSWSTEEL", "JINDALSTEL", "SAIL", "HINDALCO"],
    "JSWSTEEL":   ["TATASTEEL", "JINDALSTEL", "SAIL"],
    "JINDALSTEL": ["TATASTEEL", "JSWSTEEL", "SAIL"],
    "SAIL":       ["TATASTEEL", "JSWSTEEL", "JINDALSTEL"],
    "HINDALCO":   ["VEDL", "JSWSTEEL", "TATASTEEL", "NMDC"],
    "VEDL":       ["HINDALCO", "NMDC", "COALINDIA"],
    "NMDC":       ["VEDL", "HINDALCO", "COALINDIA"],
    "COALINDIA":  ["NMDC", "VEDL"],

    # ===== FMCG =====
    "HINDUNILVR": ["ITC", "NESTLEIND", "DABUR", "MARICO"],
    "ITC":        ["HINDUNILVR", "NESTLEIND", "DABUR", "GODREJCP"],
    "NESTLEIND":  ["HINDUNILVR", "BRITANNIA", "MARICO", "DABUR"],
    "BRITANNIA":  ["NESTLEIND", "HINDUNILVR", "MARICO", "DABUR"],
    "DABUR":      ["HINDUNILVR", "MARICO", "GODREJCP", "COLPAL"],
    "MARICO":     ["DABUR", "HINDUNILVR", "GODREJCP", "COLPAL"],
    "GODREJCP":   ["DABUR", "MARICO", "HINDUNILVR", "COLPAL"],
    "COLPAL":     ["DABUR", "MARICO", "HINDUNILVR"],
    "TATACONSUM": ["HINDUNILVR", "NESTLEIND", "DABUR"],
    "UNITDSPR":   ["UBL", "HINDUNILVR"],

    # ===== Cement =====
    "ULTRACEMCO": ["AMBUJACEM", "SHREECEM", "ACC"],
    "AMBUJACEM":  ["ULTRACEMCO", "ACC"],

    # ===== Pharma =====
    "SUNPHARMA":  ["DRREDDY", "CIPLA", "LUPIN", "TORNTPHARM"],
    "DRREDDY":    ["SUNPHARMA", "CIPLA", "LUPIN", "TORNTPHARM"],
    "CIPLA":      ["SUNPHARMA", "DRREDDY", "LUPIN", "TORNTPHARM"],
    "LUPIN":      ["SUNPHARMA", "CIPLA", "AUROPHARMA", "TORNTPHARM"],
    "AUROPHARMA": ["LUPIN", "CIPLA", "TORNTPHARM", "BIOCON"],
    "TORNTPHARM": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN"],
    "DIVISLAB":   ["SUNPHARMA", "DRREDDY", "BIOCON"],
    "BIOCON":     ["LUPIN", "AUROPHARMA", "DIVISLAB"],
    "APOLLOHOSP": ["FORTIS", "MAXHEALTH"],

    # ===== Telecom =====
    "BHARTIARTL": ["IDEA"],
    "IDEA":       ["BHARTIARTL"],

    # ===== NBFC / Financial Services =====
    "BAJFINANCE": ["BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN", "MUTHOOTFIN"],
    "BAJAJFINSV": ["BAJFINANCE", "CHOLAFIN", "SHRIRAMFIN"],
    "CHOLAFIN":   ["BAJFINANCE", "SHRIRAMFIN", "MUTHOOTFIN"],
    "SHRIRAMFIN": ["BAJFINANCE", "CHOLAFIN", "MUTHOOTFIN"],
    "MUTHOOTFIN": ["BAJFINANCE", "SHRIRAMFIN", "CHOLAFIN"],
    "SBICARD":    ["BAJFINANCE", "BAJAJFINSV"],
    "BAJAJHLDNG": ["BAJFINANCE", "BAJAJFINSV"],
    "RECLTD":     ["PFC", "IRFC"],
    "PFC":        ["RECLTD", "IRFC"],
    "IRFC":       ["RECLTD", "PFC"],

    # ===== Insurance =====
    "HDFCLIFE":   ["SBILIFE", "ICICIPRULI", "LICI"],
    "SBILIFE":    ["HDFCLIFE", "ICICIPRULI", "LICI"],
    "ICICIPRULI": ["HDFCLIFE", "SBILIFE", "LICI"],
    "ICICIGI":    ["HDFCLIFE", "SBILIFE"],
    "LICI":       ["HDFCLIFE", "SBILIFE", "ICICIPRULI"],

    # ===== Conglomerates / Diversified =====
    "LT":         ["SIEMENS", "ABB", "CUMMINSIND"],
    "SIEMENS":    ["LT", "ABB", "CUMMINSIND"],
    "ABB":        ["LT", "SIEMENS", "CUMMINSIND"],
    "CUMMINSIND": ["LT", "SIEMENS", "ABB"],
    "BHEL":       ["LT", "SIEMENS", "ABB"],

    # ===== Defence / Aerospace =====
    "HAL":        ["BEL", "BDL", "MAZDOCK"],
    "BEL":        ["HAL", "BDL", "MAZDOCK"],
    "BDL":        ["HAL", "BEL", "MAZDOCK"],
    "MAZDOCK":    ["HAL", "BEL", "COCHINSHIP"],
    "COCHINSHIP": ["MAZDOCK"],

    # ===== Retail / E-commerce / New-Age =====
    "DMART":      ["TRENT", "ETERNAL", "NYKAA"],
    "TRENT":      ["DMART", "ETERNAL"],
    "ETERNAL":    ["DMART", "TRENT", "NYKAA"],
    "NYKAA":      ["DMART", "TRENT", "ETERNAL"],
    "PAYTM":      ["POLICYBZR"],
    "POLICYBZR":  ["PAYTM"],
    "NAUKRI":     ["ETERNAL", "PAYTM"],

    # ===== Realty =====
    "DLF":        ["GODREJPROP", "OBEROIRLTY", "LODHA"],
    "GODREJPROP": ["DLF", "OBEROIRLTY", "LODHA"],
    "OBEROIRLTY": ["DLF", "GODREJPROP", "LODHA"],
    "LODHA":      ["DLF", "GODREJPROP", "OBEROIRLTY"],

    # ===== Hotels & Travel =====
    "INDHOTEL":   ["INDIGO"],
    "INDIGO":     ["INDHOTEL"],
    "IRCTC":      ["INDIGO"],

    # ===== Paints =====
    "ASIANPAINT": ["BERGEPAINT", "PIDILITIND"],
    "BERGEPAINT": ["ASIANPAINT", "PIDILITIND"],
    "PIDILITIND": ["ASIANPAINT", "BERGEPAINT"],

    # ===== Consumer Durables =====
    "TITAN":      ["DMART", "TRENT"],
    "HAVELLS":    ["SIEMENS", "ABB"],

    # ===== Chemicals / Agri =====
    "UPL":        ["PIIND"],
    "PIIND":      ["UPL"],

    # ===== Adani group =====
    "ADANIENT":   ["ADANIPORTS", "ADANIPOWER", "ADANIGREEN"],
    "ADANIPORTS": ["ADANIENT"],

    # ===== Misc =====
    "GRASIM":     ["ULTRACEMCO", "AMBUJACEM", "HINDALCO"],
    "PGHH":       ["HINDUNILVR", "MARICO", "DABUR"],
    "RVNL":       ["IRCTC", "IRFC"],
    "GMRAIRPORT": ["IRCTC"],
}


def get_peers(symbol: str) -> list[str]:
    """
    Return the list of peer symbols for a given stock.
    Returns empty list if symbol not in map (caller should fall back to
    sector-based lookup via Yahoo Finance).
    """
    return PEER_MAP.get(symbol.upper(), [])
