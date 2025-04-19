#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import csv
import time
import datetime

# Add ccxt library path
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root + '/python')

import ccxt  # noqa: E402

def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    """
    Retry fetching OHLCV data
    """
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return ohlcv
    except Exception as e:
        if num_retries > max_retries:
            print(f"Failed to fetch data: {e}")
            raise
        print(f"Retrying to fetch data ({num_retries}/{max_retries})...")
        time.sleep(exchange.rateLimit / 1000)  # Sleep before retrying
        return retry_fetch_ohlcv(exchange, max_retries - 1, symbol, timeframe, since, limit)

def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit, end_time=None):
    """
    Scrape OHLCV data until current time or end time
    """
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    
    # If no end time specified, use current time
    if end_time is None:
        end_time = exchange.milliseconds()
    
    all_ohlcv = []
    fetch_since = since
    
    while fetch_since < end_time:
        print(f"Fetching {symbol} {timeframe} data from {exchange.iso8601(fetch_since)}")
        ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
        
        # If no data received, move time window and continue
        if len(ohlcv) == 0:
            fetch_since = fetch_since + timedelta
            continue
            
        # Update next fetch start time
        fetch_since = ohlcv[-1][0] + 1
        
        all_ohlcv.extend(ohlcv)
        print(f"Retrieved {len(all_ohlcv)} candlesticks from {exchange.iso8601(all_ohlcv[0][0])} to {exchange.iso8601(all_ohlcv[-1][0])}")
        
        # Avoid too frequent requests
        time.sleep(exchange.rateLimit / 1000)
    
    # Filter results to ensure within specified time range
    return exchange.filter_by_since_limit(all_ohlcv, since, end_time, key=0)

def write_to_csv(filename, data, exchange):
    """
    Write data to CSV file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, mode='w', newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        csv_writer.writerow(['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume'])
        
        # Write data
        for candle in data:
            timestamp = candle[0]
            datetime_str = exchange.iso8601(timestamp)
            row = [timestamp, datetime_str] + candle[1:]
            csv_writer.writerow(row)
    
    print(f"Saved {len(data)} candlesticks to {filename}")

def download_crypto_data(exchange_id, symbols, timeframes, start_dates, end_dates, output_dir='data'):
    """
    Download cryptocurrency data
    
    Parameters:
    exchange_id - Exchange ID, e.g., 'binance'
    symbols - List of trading pairs, e.g., ['BTC/USDT', 'ETH/USDT']
    timeframes - List of time periods, e.g., ['1d', '1h']
    start_dates - List of start dates corresponding to timeframes, e.g., ['2017-01-01T00:00:00Z', '2021-01-01T00:00:00Z']
    end_dates - List of end dates corresponding to timeframes, e.g., [None, '2025-02-28T23:59:59Z']
    output_dir - Output directory
    """
    # Initialize exchange
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,  # Enable request rate limiting
    })
    
    # Load market data
    exchange.load_markets()
    
    # Set maximum number of candlesticks per request
    limit = 1000  # Most exchanges limit to 1000
    max_retries = 3
    
    for symbol in symbols:
        symbol_file_name = symbol.replace('/', '_')
        
        for i, timeframe in enumerate(timeframes):
            # Convert date strings to millisecond timestamps
            since = exchange.parse8601(start_dates[i])
            end_time = None if end_dates[i] is None else exchange.parse8601(end_dates[i])
            
            # Create output filename
            filename = f"{output_dir}/{exchange_id}/{symbol_file_name}_{timeframe}.csv"
            
            print(f"\nStarting to download {exchange_id} {symbol} {timeframe} data...")
            print(f"Time range: {start_dates[i]} to {end_dates[i] if end_dates[i] else 'now'}")
            
            try:
                # Fetch candlestick data
                ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit, end_time)
                
                # Save to CSV
                write_to_csv(filename, ohlcv, exchange)
                
                print(f"Successfully downloaded {symbol} {timeframe} data")
            except Exception as e:
                print(f"Failed to download {symbol} {timeframe} data: {e}")
            
            # Add delay between different trading pairs and timeframes to avoid rate limiting
            time.sleep(1)

def main():
    # Exchange ID
    exchange_id = 'binance'
    
    # Trading pairs
    symbols = ['BTC/USDT']
    
    # Time periods
    timeframes = ['1d', '1h']
    
    # Start dates (corresponding to each timeframe)
    # Daily data from 2019 (when binance.us started operations)
    # Hourly data from January 1, 2021
    start_dates = ['2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z']
    
    # End dates (corresponding to each timeframe)
    # Daily data until now
    # Hourly data until 3.31, 2025
    end_dates = [None, '2025-03-31T23:59:59Z']
    
    # Output directory
    output_dir = 'crypto_data'
    
    # Download data
    download_crypto_data(exchange_id, symbols, timeframes, start_dates, end_dates, output_dir)

if __name__ == '__main__':
    main() 