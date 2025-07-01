import google.generativeai as genai
import json
import os
from datetime import datetime
from difflib import SequenceMatcher
import re
import asyncio
from playwright.async_api import async_playwright
import time
from urllib.parse import quote_plus
import logging
from datetime import datetime
import google.generativeai as genai
from numpy import dot
from numpy.linalg import norm

SIMILARITY_THRESHOLD = 0.85  # Increase threshold for better distinction
CACHE_FILE = "query_cache.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = "AIzaSyAxugT26Z-vA7stVcaXmIgG9abn4f12RXY"  
CACHE_FILE = "query_cache.json"
SIMILARITY_THRESHOLD = 0.7  # Adjust this value (0.0 to 1.0)
MAX_PAGES_TO_SCRAPE = 5
SCRAPE_TIMEOUT = 30000  # 30 seconds timeout per page

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

class WebScraper:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
    
    async def initialize(self):
        """Initialize Playwright browser."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            self.context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            logger.info("✅ Playwright browser initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}")
            raise
    
    async def close(self):
        """Close browser and playwright."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("✅ Browser closed")
        except Exception as e:
            logger.error(f"❌ Error closing browser: {e}")
    
    async def scrape_page_content(self, url):
        """Scrape content from a single webpage."""
        try:
            page = await self.context.new_page()
            await page.goto(url, timeout=SCRAPE_TIMEOUT)
            await page.wait_for_load_state('domcontentloaded', timeout=10000)
            
            # Extract main content (try multiple selectors)
            content_selectors = [
                'article',
                '[role="main"]',
                'main',
                '.content',
                '.post-content',
                '.entry-content',
                '.article-content',
                'body'
            ]
            
            content = ""
            for selector in content_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        for element in elements[:1]:  # Take first match
                            text = await element.inner_text()
                            if len(text) > len(content):
                                content = text
                        break
                except:
                    continue
            
            # Clean up content
            content = ' '.join(content.split())[:5000]  # Limit to 5000 chars
            
            # Get page title
            title = await page.title()
            
            await page.close()
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to scrape {url}: {e}")
            return {
                'url': url,
                'title': 'Failed to load',
                'content': '',
                'success': False
            }
    
    def _is_valid_url(self, url):
        """Check if URL is valid and not from search engine domains."""
        if not url or not url.startswith('http'):
            return False
        
        # Skip search engine and unwanted domains
        skip_domains = [
            'google.com', 'google.co', 'googleusercontent.com',
            'duckduckgo.com', 'bing.com', 'yahoo.com',
            'facebook.com', 'twitter.com', 'linkedin.com/search',
            'youtube.com/results', 'maps.google.com'
        ]
        
        url_lower = url.lower()
        for domain in skip_domains:
            if domain in url_lower:
                return False
        
        return True
    
    async def search_bing(self, query):
        """Search Bing as additional fallback."""
        try:
            page = await self.context.new_page()
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
            
            await page.goto(search_url, timeout=SCRAPE_TIMEOUT)
            await page.wait_for_load_state('domcontentloaded', timeout=20000)
            await asyncio.sleep(2)
            
            # Bing selectors
            selectors_to_try = [
                'h2 a[href^="http"]',
                'li.b_algo h2 a',
                'a[href*="://"][href*="."]'
            ]
            
            urls = []
            
            for selector in selectors_to_try:
                try:
                    elements = await page.query_selector_all(selector)
                    logger.info(f"Found {len(elements)} elements with Bing selector: {selector}")
                    
                    for element in elements:
                        try:
                            href = await element.get_attribute('href')
                            if href:
                                urls.append(href)
                                if len(urls) >= MAX_PAGES_TO_SCRAPE:
                                    break
                        except:
                            continue
                    
                    if urls:
                        break
                except Exception as e:
                    logger.debug(f"Bing selector {selector} failed: {e}")
                    continue
            
            unique_urls = []
            seen = set()
            for url in urls:
                if url not in seen:
                    unique_urls.append(url)
                    seen.add(url)
            
            await page.close()
            logger.info(f"✅ Found {len(unique_urls)} Bing search results")
            return unique_urls[:MAX_PAGES_TO_SCRAPE + 1]
            
        except Exception as e:
            logger.error(f"❌ Bing search failed: {e}")
            return []
    async def search_and_scrape(self, query):
        """Main function to search and scrape top results with multiple fallbacks."""
        logger.info(f"🔍 Starting web search for: {query}")
        
        urls = []
        
        search_engines = [
            ("Bing", self.search_bing)
        ]
        
        for engine_name, search_func in search_engines:
            try:
                logger.info(f"🔍 Trying {engine_name}...")
                urls = await search_func(query)
                if urls:
                    logger.info(f"✅ {engine_name} found {len(urls)} results")
                    break
                else:
                    logger.info(f"❌ {engine_name} found no results")
            except Exception as e:
                logger.error(f"❌ {engine_name} failed: {e}")
                continue
        
        # If still no URLs, try a direct approach with basic search
        if not urls:
            logger.info("🔄 Trying direct search approach...")
            urls = await self._direct_search(query)
        
        if not urls:
            logger.error("❌ No search results found from any engine")
            return []
        
        logger.info(f"📄 Scraping {len(urls)} pages...")
        
        # Scrape all pages concurrently
        tasks = [self.scrape_page_content(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        for result in results:
            if isinstance(result, dict) and result.get('success') and result.get('content'):
                successful_results.append(result)
        
        logger.info(f"✅ Successfully scraped {len(successful_results)} pages")
        return successful_results
    
    async def _direct_search(self, query):
        """Direct search approach as last resort."""
        try:
            # Try searching with a simple approach
            page = await self.context.new_page()
            
            # Try Google with different parameters
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en"
            await page.goto(search_url, timeout=SCRAPE_TIMEOUT)
            await page.wait_for_load_state('domcontentloaded')
            await asyncio.sleep(3)
            
            # Get page content and extract URLs manually
            content = await page.content()
            urls = []
            
            # Use regex to find URLs in the page content
            import re
            url_pattern = r'href="(https?://[^"]+)"'
            matches = re.findall(url_pattern, content)
            
            for match in matches:
                # if self._is_valid_url(match):
                    urls.append(match)
                    if len(urls) >= MAX_PAGES_TO_SCRAPE:
                        break
            
            await page.close()
            
            # Remove duplicates
            unique_urls = list(dict.fromkeys(urls))
            logger.info(f"✅ Direct search found {len(unique_urls)} URLs")
            return unique_urls[:MAX_PAGES_TO_SCRAPE]
            
        except Exception as e:
            logger.error(f"❌ Direct search failed: {e}")
            return []

class QueryCache:
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Cache file is empty for now!")
                return {}
        return {}

    def save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving cache: {e}")

    def get_query_embedding(self, query):
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            return result["embedding"]  # ✅ access the dict key
        except Exception as e:
            print(f"❌ Embedding generation error: {e}")
            return None


    def cosine_similarity(self, vec1, vec2):
        try:
            return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        except:
            return 0.0

    def find_similar_query(self, query):
        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for cached_query, data in self.cache.items():
            cached_embedding = data.get('embedding')
            if not cached_embedding:
                continue
            similarity = self.cosine_similarity(query_embedding, cached_embedding)
            if similarity > best_similarity and similarity >= SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_match = cached_query

        return best_match, best_similarity

    def get_cached_response(self, query):
        similar_query, similarity = self.find_similar_query(query)
        if similar_query:
            cached_data = self.cache[similar_query]
            return {
                'response': cached_data['response'],
                'original_query': similar_query,
                'similarity': similarity,
                'cached_at': cached_data['timestamp'],
                'sources': cached_data.get('sources', [])
            }
        return None

    def cache_response(self, query, response, sources=None):
        embedding = self.get_query_embedding(query)
        if embedding is None:
            print(f"⚠️ Failed to cache due to missing embedding for: {query}")
            return

        self.cache[query] = {
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'sources': sources or [],
            'embedding': embedding
        }
        self.save_cache()

def is_informational_query(query: str) -> bool:
    """Let Gemini classify if the query is informational/search-like."""
    prompt = f"""Classify the following user input:

"{query}"

Is this a general informational query suitable for web search (e.g. 'best books on AI', 'how does Bitcoin work')?

Reply with only 'YES' or 'NO'."""
    
    try:
        response = model.generate_content(prompt)
        return "yes" in response.text.strip().lower()
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return False

def summarize_content(query, scraped_data):
    """Use Gemini to summarize scraped content."""
    if not scraped_data:
        return "No content found to summarize.", []
    
    # Prepare content for summarization
    content_text = ""
    sources = []
    
    for i, data in enumerate(scraped_data, 1):
        content_text += f"\n--- Source {i}: {data['title']} ({data['url']}) ---\n"
        content_text += data['content'][:2000]  # Limit each source to 2000 chars
        content_text += "\n\n"
        
        sources.append({
            'title': data['title'],
            'url': data['url']
        })
    
    prompt = f"""Based on the following web search results for the query "{query}", provide a comprehensive and well-structured summary:

{content_text}

Query: {query}"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), sources
    except Exception as e:
        logger.error(f"❌ Summarization error: {e}")
        return "Error generating summary from scraped content.", sources

async def main():
    """Main application loop."""
    print("🤖 Gemini-Powered Query Classifier + Web Scraper with Caching")
    print("💡 Tip: Similar queries will return cached results for faster responses!")
    print("🌐 New queries will be searched on the web and summarized!")
    print(f"📁 Cache file: {CACHE_FILE}")
    
    # Initialize components
    cache_manager = QueryCache()
    scraper = WebScraper()
    
    try:
        await scraper.initialize()
        
        while True:
            query = input("\n🔍 Enter your query (or 'quit'): ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Exiting.")
                break
            
            if not query:
                continue
            
            # Check if it's an informational query
            print("🤔 Checking if query is informational...")
            if not is_informational_query(query):
                print("⚠️ This doesn't seem like an informational query. Please try something like 'how does inflation work'.")
                continue
            
            # Check for similar cached queries
            print("🔍 Checking for similar past queries...")
            cached_result = cache_manager.get_cached_response(query)
            
            if cached_result:
                print(f"✅ Found similar query: '{cached_result['original_query']}'")
                print(f"📊 Similarity: {cached_result['similarity']:.2%}")
                print(f"⏰ Cached at: {cached_result['cached_at']}")
                
                if cached_result.get('sources'):
                    print("📚 Sources:")
                    for i, source in enumerate(cached_result['sources'], 1):
                        print(f"   {i}. {source['title']} - {source['url']}")
                
                print("\n📄 Cached Response:\n", cached_result['response'])
                
                # Ask if user wants fresh response
                refresh = input("\n🔄 Want a fresh web search instead? (y/N): ").strip().lower()
                if refresh != 'y':
                    continue
            
            # Perform web search and scraping
            try:
                print("🌐 Searching the web and scraping content...")
                scraped_data = await scraper.search_and_scrape(query)
                
                if not scraped_data:
                    print("❌ No content could be scraped from search results.")
                    continue
                
                print("🤖 Generating summary from scraped content...")
                summary, sources = summarize_content(query, scraped_data)
                
                print("\n📚 Sources:")
                for i, source in enumerate(sources, 1):
                    print(f"   {i}. {source['title']} - {source['url']}")
                
                print("\n📄 Summary:\n", summary)
                
                # Cache the new response
                cache_manager.cache_response(query, summary, sources)
                print("💾 Response cached for future similar queries!")
                
            except Exception as e:
                print(f"❌ Error during web scraping: {e}")
                logger.error(f"Web scraping error: {e}")
    
    finally:
        await scraper.close()
        print(f"\n📊 Total cached queries: {len(cache_manager.cache)}")

if __name__ == "__main__":
    # Install required packages message
    print("📦 Make sure you have installed: pip install playwright google-generativeai")
    print("🎭 Also run: playwright install chromium")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")
        logger.error(f"Application error: {e}")