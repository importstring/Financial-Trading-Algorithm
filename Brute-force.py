import aiohttp
import asyncio
import logging
import time
from bs4 import BeautifulSoup
from typing import Set, List
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    filename='yh_get_all_sym.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

HEADERS = {
    "authority": "finance.yahoo.com",
    "accept": "text/html",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
}

async def fetch_url(session: aiohttp.ClientSession, url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            async with session.get(url, headers=HEADERS) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1)

def extract_count(body: str, search_term: str) -> int:
    try:
        start = body.find('Stocks (') + 8
        end = body.find(')', start)
        count = int(body[start:end])
        logging.info(f'Count for {search_term}: {count}')
        return count
    except (ValueError, TypeError) as e:
        logging.error(f"Failed to extract count for {search_term}: {e}")
        return 0

async def process_block(session: aiohttp.ClientSession, search_term: str, block: int) -> Set[str]:
    url = f"https://finance.yahoo.com/lookup/equity?s={search_term}&t=A&b={block}&c=100"
    symbols = set()
    
    try:
        body = await fetch_url(session, url)
        soup = BeautifulSoup(body, 'html.parser')
        table = soup.find('table', {'class': 'W(100%)'})
        if table:
            rows = table.find_all('tr')
            for row in rows:
                symbol_cell = row.find('td', {'aria-label': 'Symbol'})
                if symbol_cell:
                    symbol = symbol_cell.text.strip()
                    if symbol:
                        symbols.add(symbol)
    except Exception as e:
        logging.error(f"Error processing block {block} for {search_term}: {e}")
    
    return symbols

async def process_search_term(session: aiohttp.ClientSession, search_term: str) -> Set[str]:
    url = f"https://finance.yahoo.com/lookup/equity?s={search_term}&t=A&b=0&c=25"
    symbols = set()
    
    try:
        body = await fetch_url(session, url)
        count = extract_count(body, search_term)
        
        if count == 0:
            return symbols
            
        tasks = []
        for block in range(0, min(count, 9000), 100):
            tasks.append(process_block(session, search_term, block))
            
        results = await asyncio.gather(*tasks)
        for result in results:
            symbols.update(result)
            
    except Exception as e:
        logging.error(f"Error processing search term {search_term}: {e}")
        
    return symbols

async def main():
    search_chars = [chr(x) for x in range(65, 91)] + [chr(x) for x in range(48, 58)]
    all_symbols = set()
    
    async with aiohttp.ClientSession() as session:
        for c1 in tqdm(search_chars, desc="Primary characters"):
            for c2 in search_chars:
                search_term = c1 + c2
                symbols = await process_search_term(session, search_term)
                
                if len(symbols) >= 9000:
                    for c3 in search_chars:
                        search_term = c1 + c2 + c3
                        additional_symbols = await process_search_term(session, search_term)
                        symbols.update(additional_symbols)
                
                all_symbols.update(symbols)
                logging.info(f"Total symbols so far: {len(all_symbols)}")

    output_file = Path("yh_all_symbols.txt")
    output_file.write_text(str(all_symbols), encoding='utf-8')
    print(f"Total symbols collected: {len(all_symbols)}")

if __name__ == '__main__':
    asyncio.run(main())
