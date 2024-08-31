import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import logging
import random
import os

class WebScrapper:

    def __init__(self, base_url, search_query, filename, proxies, mac_address):
        self.base_url = base_url
        self.search_query = search_query
        self.filename = filename
        self.proxies = proxies
        self.mac_address = mac_address
        self.list_article_url = []

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_random_proxy(self):
        return random.choice(self.proxies)

    def make_request(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'X-Client-MAC': self.mac_address
        }
        proxy = self.get_random_proxy()
        try:
            response = requests.get(url, headers=headers, proxies={'http': proxy, 'https': proxy})
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logging.error(f"Error fetching content from {url}: {e}")
            return None

    def scrape_articles(self):
        search_url = f"{self.base_url}/search?q={self.search_query}"
        response = self.make_request(search_url)
        if not response:
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            parsed_url = urlparse(href)
            if parsed_url.scheme in ['http', 'https']:
                self.list_article_url.append(href)
            elif parsed_url.path == '/url':
                query_params = parse_qs(parsed_url.query)
                if 'url' in query_params:
                    article_url = query_params['url'][0]
                    self.list_article_url.append(article_url)
        
        logging.info(f"Found {len(self.list_article_url)} articles.")

    def _get_article_content(self, article_url):
        response = self.make_request(article_url)
        if not response:
            return None, None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('h1').text if soup.find('h1') else 'No Title'
        content = soup.find('main') or soup.find('div', id='main')
        if content:
            content = content.get_text(separator='\n')
        else:
            content = 'No Content'
        
        return title, content

    def save_to_markdown(self):
        with open(self.filename, 'w', encoding='utf-8') as file:
            for article_url in self.list_article_url:
                title, content = self._get_article_content(article_url)
                if title and content:
                    file.write(f"## {title}\n\n{content}\nRead more\n")
                else:
                    logging.warning(f"Skipping article {article_url} due to missing content.")
        
        logging.info(f"Articles saved to {self.filename}")

if __name__ == "__main__":
    base_url = "https://www.google.com"
    search_query = "Generative%20adversarial%20network"
    filename = "articles.md"
    proxies = [os.environ.get('proxy1'),os.environ.get('proxy2'),os.environ.get('proxy3')]
    mac_address = os.environ.get('mac')
    w = WebScrapper(base_url, search_query, filename, proxies, mac_address)
    w.scrape_articles()
    w.save_to_markdown()
