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

class TDSDataScraperAdvanced:
    def __init__(self, cookies_file="discourse_cookies.json", max_depth=3):
        self.cookies_file = cookies_file
        self.max_depth = max_depth
        self.setup_driver()
        self.course_data = []
        self.discourse_data = []
        self.visited_urls = set()
        self.url_queue = deque()

    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        chrome_options.add_argument("--enable-javascript")
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 15)

    def load_cookies(self, domain):
        if os.path.exists(self.cookies_file):
            with open(self.cookies_file, 'r') as f:
                cookies = json.load(f)

            self.driver.get(f"https://{domain}")
            time.sleep(2)

            for cookie in cookies:
                try:
                    if 'domain' in cookie and domain in cookie['domain']:
                        self.driver.add_cookie(cookie)
                except Exception as e:
                    continue

            self.driver.refresh()
            time.sleep(3)

    def wait_for_js_load(self, timeout=10):
        self.driver.execute_script("return document.readyState") == "complete"
        time.sleep(2)

        try:
            WebDriverWait(self.driver, timeout).until(
                lambda driver: driver.execute_script("return jQuery.active == 0") if driver.execute_script("return typeof jQuery != 'undefined'") else True
            )
        except:
            pass

        time.sleep(1)

    def extract_links(self, base_url, current_depth):
        links = set()
        try:
            link_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href]")

            for element in link_elements:
                try:
                    href = element.get_attribute('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        parsed_url = urlparse(full_url)

                        if (parsed_url.netloc in ['tds.s-anand.net', 'discourse.onlinedegree.iitm.ac.in'] and
                            full_url not in self.visited_urls and
                            current_depth < self.max_depth and
                            not any(ext in full_url.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip', '.doc'])):
                            links.add(full_url)
                except:
                    continue

        except Exception as e:
            print(f"Error extracting links: {e}")

        return links

    def click_interactive_elements(self):
        clickable_selectors = [
            "button", ".btn", "[onclick]", ".clickable", ".toggle",
            "[role='button']", ".accordion", ".tab", ".menu-item",
            ".nav-link", ".dropdown", "[data-toggle]"
        ]

        for selector in clickable_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements[:5]:
                    try:
                        if element.is_displayed() and element.is_enabled():
                            self.driver.execute_script("arguments[0].click();", element)
                            time.sleep(1)
                            self.wait_for_js_load(5)
                    except:
                        continue
            except:
                continue

    def scrape_course_content_deep(self):
        print("Starting deep course content scraping...")

        start_url = "https://tds.s-anand.net/#/2025-01/"
        self.url_queue.append((start_url, 0))

        while self.url_queue:
            url, depth = self.url_queue.popleft()

            if url in self.visited_urls or depth > self.max_depth:
                continue

            print(f"Scraping course URL (depth {depth}): {url}")
            self.visited_urls.add(url)

            try:
                self.driver.get(url)
                self.wait_for_js_load()

                self.click_interactive_elements()

                try:
                    course_title = self.driver.find_element(By.CSS_SELECTOR, "h1, .title, .course-title, .page-title").text
                except:
                    course_title = f"TDS 2025-01 - Depth {depth}"

                content_selectors = [
                    ".content", ".lesson", ".module", ".chapter", ".section",
                    ".markdown", ".md", "main", "article", ".post", ".entry",
                    "[class*='content']", "[class*='lesson']", "[class*='module']",
                    ".text", ".description", ".body", "p", ".paragraph"
                ]

                page_content = []

                for selector in content_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in elements:
                            text = element.text.strip()
                            if len(text) > 30 and text not in [item['content'] for item in page_content]:
                                try:
                                    section_title = element.find_element(By.CSS_SELECTOR, "h1, h2, h3, h4, h5, .title, .heading").text
                                except:
                                    section_title = f"Section - Depth {depth}"

                                page_content.append({
                                    'source': 'course_content',
                                    'course': course_title,
                                    'section': section_title,
                                    'content': text,
                                    'url': url,
                                    'depth': depth,
                                    'scraped_at': datetime.now().isoformat()
                                })
                    except:
                        continue

                if not page_content:
                    try:
                        body_text = self.driver.find_element(By.TAG_NAME, "body").text
                        if len(body_text) > 100:
                            page_content.append({
                                'source': 'course_content',
                                'course': course_title,
                                'section': f'Full Page - Depth {depth}',
                                'content': body_text,
                                'url': url,
                                'depth': depth,
                                'scraped_at': datetime.now().isoformat()
                            })
                    except:
                        pass

                self.course_data.extend(page_content)

                if depth < self.max_depth:
                    new_links = self.extract_links(url, depth)
                    for link in new_links:
                        self.url_queue.append((link, depth + 1))

            except Exception as e:
                print(f"Error scraping course URL {url}: {e}")

    def scrape_discourse_posts_deep(self):
        print("Starting deep Discourse scraping...")

        discourse_url = "https://discourse.onlinedegree.iitm.ac.in"
        self.load_cookies("discourse.onlinedegree.iitm.ac.in")

        category_urls = [
            f"{discourse_url}/c/bsc-degree/tools-in-data-science/84",
            f"{discourse_url}/c/courses/tools-in-data-science/31",
            f"{discourse_url}/search?q=tools%20data%20science",
            f"{discourse_url}/latest"
        ]

        target_date = datetime(2025, 1, 1)
        discourse_urls_visited = set()

        for category_url in category_urls:
            url_queue = deque([(category_url, 0)])

            while url_queue:
                url, depth = url_queue.popleft()

                if url in discourse_urls_visited or depth > self.max_depth:
                    continue

                print(f"Scraping Discourse URL (depth {depth}): {url}")
                discourse_urls_visited.add(url)

                try:
                    self.driver.get(url)
                    self.wait_for_js_load()

                    self.click_interactive_elements()

                    topic_selectors = [
                        "a.title", ".topic-list-item a", ".topic-link",
                        "[href*='/t/']", ".search-result a", ".post-link"
                    ]

                    topic_links = set()
                    for selector in topic_selectors:
                        try:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            for element in elements:
                                href = element.get_attribute('href')
                                if href and '/t/' in href and href not in discourse_urls_visited:
                                    topic_links.add(href)
                        except:
                            continue

                    for topic_url in list(topic_links)[:20]:
                        try:
                            self.scrape_discourse_topic_deep(topic_url, target_date)
                            time.sleep(1)

                            if depth < self.max_depth:
                                url_queue.append((topic_url, depth + 1))

                        except Exception as e:
                            continue

                    try:
                        pagination_links = self.driver.find_elements(By.CSS_SELECTOR, ".next-page, .btn-next, .pagination a")
                        for link in pagination_links[:3]:
                            href = link.get_attribute('href')
                            if href and href not in discourse_urls_visited and depth < self.max_depth:
                                url_queue.append((href, depth + 1))
                    except:
                        pass

                except Exception as e:
                    print(f"Error scraping Discourse category {url}: {e}")

    def scrape_discourse_topic_deep(self, topic_url, target_date):
        try:
            self.driver.get(topic_url)
            self.wait_for_js_load()

            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            try:
                load_more_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".load-more, .show-more, [data-action='load-more']")
                for button in load_more_buttons:
                    try:
                        if button.is_displayed():
                            button.click()
                            self.wait_for_js_load()
                    except:
                        continue
            except:
                pass

            try:
                title = self.driver.find_element(By.CSS_SELECTOR, "h1.fancy-title, .topic-title, h1").text
            except:
                title = "TDS Discussion"

            post_selectors = [
                ".topic-post", ".post", ".cooked", ".post-content",
                "[itemtype*='Comment']", ".reply", ".message"
            ]

            for selector in post_selectors:
                try:
                    posts = self.driver.find_elements(By.CSS_SELECTOR, selector)

                    for post in posts:
                        try:
                            post_content = post.text.strip()
                            if len(post_content) < 20:
                                continue

                            try:
                                date_element = post.find_element(By.CSS_SELECTOR, ".post-date, .crawler-post-infos time, time, .date")
                                post_date_str = date_element.get_attribute('datetime') or date_element.get_attribute('title') or date_element.text
                            except:
                                post_date_str = datetime.now().isoformat()

                            try:
                                if 'T' in post_date_str:
                                    post_date = datetime.fromisoformat(post_date_str.replace('Z', '+00:00'))
                                else:
                                    try:
                                        post_date = datetime.strptime(post_date_str, '%Y-%m-%d')
                                    except:
                                        post_date = datetime.now()

                                post_date = post_date.replace(tzinfo=None)
                            except:
                                post_date = datetime.now()

                            if post_date >= target_date:
                                try:
                                    author = post.find_element(By.CSS_SELECTOR, ".username, .post-author, .author, [data-username]").text
                                except:
                                    author = "Unknown"

                                post_data = {
                                    'source': 'discourse',
                                    'title': title,
                                    'author': author,
                                    'content': post_content,
                                    'date': post_date.isoformat(),
                                    'url': topic_url,
                                    'scraped_at': datetime.now().isoformat()
                                }

                                if post_data not in self.discourse_data:
                                    self.discourse_data.append(post_data)

                        except Exception as e:
                            continue

                    if posts:
                        break

                except Exception as e:
                    continue

        except Exception as e:
            print(f"Error scraping topic {topic_url}: {e}")

    def save_data(self):
        print("Saving scraped data...")

        if self.course_data:
            course_df = pd.DataFrame(self.course_data)
            course_df.drop_duplicates(subset=['content'], inplace=True)
            course_df.to_csv('tds_course_content_deep.csv', index=False)
            course_df.to_json('tds_course_content_deep.json', orient='records', indent=2)

        if self.discourse_data:
            discourse_df = pd.DataFrame(self.discourse_data)
            discourse_df.drop_duplicates(subset=['content', 'url'], inplace=True)
            discourse_df.to_csv('tds_discourse_posts_deep.csv', index=False)
            discourse_df.to_json('tds_discourse_posts_deep.json', orient='records', indent=2)

        print(f"Scraped {len(self.course_data)} course content items across {len(self.visited_urls)} pages")
        print(f"Scraped {len(self.discourse_data)} discourse posts")

    def run(self):
        try:
            self.scrape_course_content_deep()
            self.scrape_discourse_posts_deep()
            self.save_data()
        finally:
            self.driver.quit()

if __name__ == "__main__":
    scraper = TDSDataScraperAdvanced(max_depth=3)
    scraper.run()