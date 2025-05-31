import logging
import time
from datetime import datetime
from logging.config import dictConfig
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai


load_dotenv('.env.private')

dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '{asctime} {levelname} {process} [{filename}:{lineno}] - {message}',
            'style': '{',
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
})

logger = logging.getLogger(__name__)

# https://ai.google.dev/gemini-api/docs/pricing
MODELS = {
    'gemini-2.5-flash-preview-05-20': {
        'input_token_cost': 0.15 / 1000000,
        'output_token_cost': 3.50 / 1000000,
    },
    'gemini-2.5-pro-preview-05-06': {
        'input_token_cost': 1.25 / 1000000,
        'output_token_cost': 10 / 1000000,
    },
}

ITEM_XML_TEMPLATE = """<item>
  <title>{title}</title>
  <url>{url}</url>
  <content>{content}</content>
</item>
"""

gemini = genai.Client(
    http_options=genai.types.HttpOptions(timeout=120 * 1000)
)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def done(self):
        self.latency = time.time() - self.t0


class ProgressMeter:
    def __init__(self, total, msg='{done}/{total} ({percent}%) done', mod=10):
        self.total = total
        self.done = 0
        self.msg = msg
        self.mod = mod

    def increment(self):
        self.done += 1

        if self.done % self.mod == 0:
            percent = round((self.done / self.total) * 100)
            logger.info(self.msg.format(done=self.done, total=self.total, percent=percent))


def make_prompt():
    with open('prompt.txt') as f:
        prompt = f.read()

    return prompt.format(date=datetime.now().strftime('%B %d, %Y'))


def estimate_cost(res):
    # gemini-2.5-pro-preview-05-06 appears as models/gemini-2.5-pro-preview-05-06
    model_version = res.model_version.replace('models/', '')
    input_cost = res.usage_metadata.prompt_token_count * MODELS[model_version]['input_token_cost']

    if res.usage_metadata.candidates_token_count:
        output_cost = res.usage_metadata.candidates_token_count * MODELS[model_version]['output_token_cost']
    else:
        # candidates_token_count is sometimes None, unclear why
        logger.info('no candidates_token_count, unable to estimate output cost')
        output_cost = 0

    return input_cost + output_cost


def run(with_cache=False):
    session = requests.Session()
    session.headers.update({
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/136.0.0.0 Safari/537.36'
        )
    })

    items_xml = None
    cache_path = Path('items.xml')

    if with_cache and cache_path.exists():
        items_xml = cache_path.read_text()

    if not items_xml:
        home_response = session.get('https://apnews.com/')
        home_soup = BeautifulSoup(home_response.text, 'html.parser')
        links = home_soup.select('.PagePromo-title')

        logger.info(f'found {len(links)} links')

        article_timer = Timer()
        link_progress = ProgressMeter(len(links), msg='tried {done}/{total} links ({percent}%)')

        articles = {}
        for link in links:
            try:
                a_tag = link.select_one('a')
                if not a_tag:
                    logger.info('skipping link with no <a> tag')
                    link_progress.increment()
                    continue

                link_href = a_tag.get('href')
                if '/article/' not in link_href:
                    logger.info(f'skipping non-article link: {link_href}')
                    link_progress.increment()
                    continue

                # Articles can be linked to multiple times. For example, a link can appear
                # in a regular section on the homepage and also in a "popular" section.
                # Prefer the link with the most information in the title.
                article_title = link.get_text(strip=True)
                known_article = articles.get(link_href)
                if known_article and len(known_article['title']) >= len(article_title):
                    logger.info(f'skipping known article link: {link_href}')
                    link_progress.increment()
                    continue

                logger.info(f'getting content for article link: {link_href} ({article_title})')

                article_response = session.get(link_href)
                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                p_tags = article_soup.select('.RichTextStoryBody > p')

                contents = []
                for p_tag in p_tags:
                    # Avoid stripping internal whitespace (e.g., around <a> text)
                    content = p_tag.get_text().strip()

                    # Some articles end with notes below a horizontal rule of varying length. We don't need them.
                    if content.startswith('__') or content.startswith('——'):
                        break

                    contents.append(content)

                article_content = ' '.join(contents)

                if not article_content:
                    logger.info(f'got empty content for article link: {link_href}')

                articles[link_href] = {
                    'title': article_title,
                    'url': link_href,
                    'content': article_content,
                }
            except:
                logger.exception('failed to handle link')
            finally:
                link_progress.increment()
                time.sleep(3)

        article_timer.done()
        logger.info(f'loaded {len(articles)} articles in {round(article_timer.latency, 2)}s')

        # TODO: map from full article url to short representation, then replace, to reduce mistakes
        items_xml = ''
        for article in articles.values():
            item_xml = ITEM_XML_TEMPLATE.format(
                title=article['title'],
                url=article['url'],
                content=article['content']
            )

            items_xml += item_xml

        if with_cache:
            cache_path.write_text(items_xml)

    contents = [
        genai.types.Content(
            role='user',
            parts=[
                genai.types.Part.from_text(text=items_xml),
            ]
        ),
    ]

    generation_timer = Timer()
    res = gemini.models.generate_content(
        # model='gemini-2.5-flash-preview-05-20',
        model='gemini-2.5-pro-preview-05-06',
        config=genai.types.GenerateContentConfig(
            system_instruction=make_prompt(),
            temperature=0,
        ),
        contents=contents
    )
    generation_timer.done()

    logger.info(res.text)
    logger.info(f'input tokens: {res.usage_metadata.prompt_token_count}')
    logger.info(f'output tokens: {res.usage_metadata.candidates_token_count}')
    logger.info(f'latency: {round(generation_timer.latency, 2)}s')

    cost = estimate_cost(res)
    logger.info(f'cost: ${round(cost, 4)}')


def exception_handler(exception, event, context):
    logger.error('unhandled exception:', exc_info=exception)

    # Tells Zappa not to re-raise the exception, which in turn prevents Lambda
    # from retrying invocation.
    # https://github.com/zappa/Zappa/blob/0.60.1/zappa/handler.py#L252-L255
    return True
