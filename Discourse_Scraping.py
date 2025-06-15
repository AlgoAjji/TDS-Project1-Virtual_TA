
import json
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import os
import re
from collections import deque

class DiscourseScraperFocused:
    def __init__(self, cookies_file="discourse_cookies.json", max_depth=3):
        self.cookies_file = cookies_file
        self.max_depth = max_depth
        self.setup_driver()
        self.discourse_data = []
        self.visited_urls = set()
        self.target_date = datetime(2025, 1, 1)

    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])

        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, 20)

    def load_cookies_and_verify(self):
        print("Loading cookies and verifying authentication...")

        if not os.path.exists(self.cookies_file):
            print(f"Cookie file {self.cookies_file} not found!")
            return False

        with open(self.cookies_file, 'r') as f:
            cookies = json.load(f)

        self.driver.get("https://discourse.onlinedegree.iitm.ac.in")
        time.sleep(3)

        for cookie in cookies:
            try:
                if isinstance(cookie, dict) and 'name' in cookie and 'value' in cookie:
                    cookie_dict = {
                        'name': cookie['name'],
                        'value': cookie['value'],
                        'domain': cookie.get('domain', '.discourse.onlinedegree.iitm.ac.in'),
                        'path': cookie.get('path', '/'),
                    }
                    if 'secure' in cookie:
                        cookie_dict['secure'] = cookie['secure']
                    if 'httpOnly' in cookie:
                        cookie_dict['httpOnly'] = cookie['httpOnly']

                    self.driver.add_cookie(cookie_dict)
            except Exception as e:
                print(f"Error adding cookie: {e}")
                continue

        self.driver.refresh()
        time.sleep(5)

        # Check if logged in
        try:
            user_menu = self.driver.find_element(By.CSS_SELECTOR, ".header-dropdown-toggle, .current-user, .user-menu")
            print("Successfully authenticated with Discourse!")
            return True
        except:
            print("Authentication failed - checking if we can still access content...")
            try:
                # Check if we can see any content
                self.driver.find_element(By.CSS_SELECTOR, ".topic-list, .latest-topic-list, .category-list")
                print("Can access some content without full authentication...")
                return True
            except:
                print("Cannot access Discourse content - check cookies!")
                return False

    def wait_for_discourse_load(self):
        try:
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
            time.sleep(2)

            # Wait for Discourse's Ember.js to finish loading
            self.driver.execute_script("""
                return new Promise((resolve) => {
                    if (window.Ember && window.Ember.testing) {
                        Ember.Test.waitForTestHelpers().then(resolve);
                    } else {
                        setTimeout(resolve, 2000);
                    }
                });
            """)
        except:
            time.sleep(3)

    def find_tds_topics(self):
        print("Finding TDS-related topics...")
        tds_urls = set()

        # Search strategies
        search_terms = [
            "tools data science",
            "TDS",
            "data science tools",
            "python data science",
            "data analysis tools"
        ]

        # Category URLs
        category_urls = [
            "https://discourse.onlinedegree.iitm.ac.in/c/bsc-degree/tools-in-data-science/84",
            "https://discourse.onlinedegree.iitm.ac.in/c/courses/tools-in-data-science/31",
            "https://discourse.onlinedegree.iitm.ac.in/categories"
        ]

        # Search for topics by search terms
        for term in search_terms:
            try:
                search_url = f"https://discourse.onlinedegree.iitm.ac.in/search?q={term.replace(' ', '%20')}"
                print(f"Searching: {search_url}")

                self.driver.get(search_url)
                self.wait_for_discourse_load()

                # Find search results
                result_selectors = [
                    ".fps-result .topic-title a",
                    ".search-results .topic-title a",
                    ".fps-topic a",
                    ".search-result-topic a",
                    "a[href*='/t/']"
                ]

                for selector in result_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in elements:
                            href = element.get_attribute('href')
                            if href and '/t/' in href:
                                tds_urls.add(href)
                    except:
                        continue

            except Exception as e:
                print(f"Error searching for term '{term}': {e}")

        # Browse categories
        for category_url in category_urls:
            try:
                print(f"Browsing category: {category_url}")
                self.driver.get(category_url)
                self.wait_for_discourse_load()

                # Scroll to load more topics
                for _ in range(3):
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)

                # Find topic links
                topic_selectors = [
                    ".topic-list-item .title a",
                    ".latest-topic-list-item a",
                    ".topic-title a",
                    "a.title",
                    ".topic-list tbody tr .main-link a",
                    "a[href*='/t/']"
                ]

                for selector in topic_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in elements:
                            href = element.get_attribute('href')
                            if href and '/t/' in href:
                                tds_urls.add(href)
                    except:
                        continue

            except Exception as e:
                print(f"Error browsing category {category_url}: {e}")

        # Browse latest topics for TDS-related content
        try:
            print("Browsing latest topics...")
            self.driver.get("https://discourse.onlinedegree.iitm.ac.in/latest")
            self.wait_for_discourse_load()

            # Load more topics
            for _ in range(5):
                try:
                    load_more = self.driver.find_element(By.CSS_SELECTOR, ".btn-load-more, .load-more")
                    if load_more.is_displayed():
                        load_more.click()
                        time.sleep(3)
                except:
                    break

            # Find TDS-related topics by title matching
            topic_elements = self.driver.find_elements(By.CSS_SELECTOR, ".topic-list-item .title a, a.title")
            for element in topic_elements:
                try:
                    title = element.text.lower()
                    href = element.get_attribute('href')

                    if (href and '/t/' in href and
                        any(keyword in title for keyword in ['tds', 'tools', 'data science', 'python', 'pandas', 'numpy'])):
                        tds_urls.add(href)
                except:
                    continue

        except Exception as e:
            print(f"Error browsing latest topics: {e}")

        print(f"Found {len(tds_urls)} TDS-related topic URLs")
        return list(tds_urls)

    def scrape_discourse_topic(self, topic_url):
        print(f"Scraping topic: {topic_url}")

        try:
            self.driver.get(topic_url)
            self.wait_for_discourse_load()

            # Scroll to load all posts
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            while True:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            # Get topic title
            try:
                title = self.driver.find_element(By.CSS_SELECTOR, "#topic-title, .fancy-title, h1").text.strip()
            except:
                title = "Unknown Topic"

            # Find all posts
            post_selectors = [
                ".topic-post",
                ".post-stream .boxed",
                "article[data-post-id]",
                ".post"
            ]

            posts = []
            for selector in post_selectors:
                try:
                    found_posts = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if found_posts:
                        posts = found_posts
                        break
                except:
                    continue

            print(f"Found {len(posts)} posts in topic")

            for post in posts:
                try:
                    # Extract post content
                    content_selectors = [
                        ".cooked",
                        ".post-content",
                        ".regular.contents",
                        ".topic-body"
                    ]

                    post_content = ""
                    for content_selector in content_selectors:
                        try:
                            content_element = post.find_element(By.CSS_SELECTOR, content_selector)
                            post_content = content_element.text.strip()
                            if post_content:
                                break
                        except:
                            continue

                    if len(post_content) < 10:
                        continue

                    # Extract author
                    try:
                        author_selectors = [
                            ".username a",
                            "[data-username]",
                            ".names .username",
                            ".post-author"
                        ]
                        author = "Unknown"
                        for author_selector in author_selectors:
                            try:
                                author_element = post.find_element(By.CSS_SELECTOR, author_selector)
                                author = author_element.text.strip() or author_element.get_attribute('data-username')
                                if author:
                                    break
                            except:
                                continue
                    except:
                        author = "Unknown"

                    # Extract date
                    try:
                        date_selectors = [
                            ".post-date",
                            ".crawler-post-infos time",
                            "time",
                            ".relative-date"
                        ]
                        post_date = datetime.now()
                        for date_selector in date_selectors:
                            try:
                                date_element = post.find_element(By.CSS_SELECTOR, date_selector)
                                date_str = (date_element.get_attribute('datetime') or
                                          date_element.get_attribute('title') or
                                          date_element.text)

                                if date_str:
                                    try:
                                        if 'T' in date_str:
                                            post_date = datetime.fromisoformat(date_str.replace('Z', '+00:00').replace('+00:00', ''))
                                        else:
                                            post_date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                                        break
                                    except:
                                        continue
                            except:
                                continue
                    except:
                        post_date = datetime.now()

                    # Only include posts from target date onwards
                    if post_date >= self.target_date:
                        post_data = {
                            'source': 'discourse',
                            'title': title,
                            'author': author,
                            'content': post_content,
                            'date': post_date.isoformat(),
                            'url': topic_url,
                            'scraped_at': datetime.now().isoformat()
                        }

                        # Avoid duplicates
                        if not any(existing['content'] == post_content and existing['url'] == topic_url
                                 for existing in self.discourse_data):
                            self.discourse_data.append(post_data)

                except Exception as e:
                    print(f"Error processing post: {e}")
                    continue

        except Exception as e:
            print(f"Error scraping topic {topic_url}: {e}")

    def scrape_discourse_deep(self):
        print("Starting focused Discourse scraping...")

        if not self.load_cookies_and_verify():
            print("Failed to authenticate with Discourse!")
            return

        # Find all TDS-related topics
        tds_topics = self.find_tds_topics()

        if not tds_topics:
            print("No TDS topics found!")
            return

        print(f"Scraping {len(tds_topics)} TDS topics...")

        for i, topic_url in enumerate(tds_topics):
            if topic_url in self.visited_urls:
                continue

            self.visited_urls.add(topic_url)
            print(f"Progress: {i+1}/{len(tds_topics)}")

            try:
                self.scrape_discourse_topic(topic_url)
                time.sleep(2)  # Be respectful to the server
            except Exception as e:
                print(f"Error with topic {topic_url}: {e}")
                continue

    def save_data(self):
        print("Saving Discourse data...")

        if self.discourse_data:
            discourse_df = pd.DataFrame(self.discourse_data)
            # Remove duplicates based on content and URL
            discourse_df.drop_duplicates(subset=['content', 'url'], inplace=True)
            discourse_df.to_csv('tds_discourse_posts_focused.csv', index=False)
            discourse_df.to_json('tds_discourse_posts_focused.json', orient='records', indent=2)
            print(f"Saved {len(discourse_df)} unique discourse posts")
        else:
            print("No discourse posts scraped!")

    def run(self):
        try:
            self.scrape_discourse_deep()
            self.save_data()
        finally:
            self.driver.quit()

if __name__ == "__main__":
    scraper = DiscourseScraperFocused(max_depth=3)
    scraper.run()
