import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from logging.config import dictConfig
from pathlib import Path

import mistletoe
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from slack_sdk import WebClient


load_dotenv('.env.private')

SLACK_BOT_TOKEN = os.environ['SLACK_BOT_TOKEN']
SLACK_CHANNEL_ID = os.environ['SLACK_CHANNEL_ID']
READ_ITEM_CACHE = bool(int(os.environ.get('READ_ITEM_CACHE', 0)))
WRITE_ITEM_CACHE = bool(int(os.environ.get('WRITE_ITEM_CACHE', 0)))

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

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/136.0.0.0 Safari/537.36'
    )
}

# https://ai.google.dev/gemini-api/docs/pricing
MODELS = {
    # <= 200k input tokens
    'gemini-2.5-pro': {
        'input_token_cost': 1.25 / 1000000,
        'output_token_cost': 10 / 1000000,
    },
    'gemini-2.5-flash': {
        'input_token_cost': 0.30 / 1000000,
        'output_token_cost': 2.50 / 1000000,
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

slack = WebClient(token=SLACK_BOT_TOKEN)


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


def get_ap_items():
    session = requests.Session()
    session.headers.update(HEADERS)

    home_response = session.get('https://apnews.com/')
    home_soup = BeautifulSoup(home_response.text, 'lxml')
    links = home_soup.select('.PagePromo-title')

    logger.info(f'found {len(links)} AP links')

    item_timer = Timer()
    link_progress = ProgressMeter(len(links), msg='tried {done}/{total} AP links ({percent}%)')

    items = {}
    for link in links:
        try:
            a_tag = link.select_one('a')
            if not a_tag:
                logger.info('skipping AP link with no <a> tag')
                continue

            link_href = a_tag.get('href')
            if '/article/' not in link_href:
                logger.info(f'skipping non-article AP link: {link_href}')
                continue

            # Articles can be linked to multiple times. For example, a link can appear
            # in a regular section on the homepage and also in a "popular" section.
            # Prefer the link with the most information in the title.
            article_title = link.get_text(strip=True)
            known_item = items.get(link_href)
            if known_item and len(known_item['title']) >= len(article_title):
                logger.info(f'skipping known AP article link: {link_href}')
                continue

            logger.info(f'getting content for AP article link: {link_href} ({article_title})')

            article_response = session.get(link_href)
            article_soup = BeautifulSoup(article_response.text, 'lxml')
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
                logger.info(f'got empty content for AP article link: {link_href}')

            items[link_href] = {
                'title': article_title,
                # TODO: use short urls, resolve to full urls during rendering?
                'url': link_href,
                'content': article_content,
            }
        except:
            logger.exception('failed to handle AP link')
        finally:
            link_progress.increment()
            time.sleep(1)

    item_timer.done()
    logger.info(f'loaded {len(items)} AP items in {round(item_timer.latency, 2)}s')

    return items


def get_nhk_items():
    session = requests.Session()
    session.headers.update(HEADERS)

    origin = 'https://www3.nhk.or.jp'
    all_response = session.get(f'{origin}/nhkworld/data/en/news/all.json')
    all_data = all_response.json()['data']

    fresh_articles = []
    now_ms = round(time.time() * 1000)
    for article in all_data:
        if now_ms - int(article['updated_at']) > (24 * 60 * 60 * 1000):
            article_path = article['page_url']
            logger.info(f'skipping stale NHK article: {origin}{article_path}')
            continue

        fresh_articles.append(article)

    logger.info(f'found {len(fresh_articles)} fresh NHK articles')

    items = {}
    item_timer = Timer()
    item_progress = ProgressMeter(len(fresh_articles), msg='tried {done}/{total} NHK items ({percent}%)')

    for article in fresh_articles:
        try:
            article_id = article['id']
            logger.info(f'getting content for NHK article {article_id}')

            detail_response = session.get(f'{origin}/nhkworld/data/en/news/{article_id}.json')
            detail_data = detail_response.json()['data']

            article_path = detail_data['page_url']
            item_url = f'{origin}{article_path}'
            detail_soup = BeautifulSoup(detail_data['detail'], 'lxml')

            items[item_url] = {
                'title': detail_data['title'],
                'url': item_url,
                'content': detail_soup.get_text(),
            }
        except:
            logger.exception('failed to handle NHK article')
        finally:
            item_progress.increment()
            time.sleep(1)

    item_timer.done()
    logger.info(f'loaded {len(items)} NHK items in {round(item_timer.latency, 2)}s')

    return items


def get_date():
    return datetime.now().strftime('%A, %B %-d, %Y')


def test_item(item):
    with open('prompts/filter.txt') as f:
        prompt_template = f.read()

    prompt = prompt_template.format(title=item['title'])
    contents = [
        genai.types.Content(
            role='user',
            parts=[
                genai.types.Part.from_text(text=prompt),
            ]
        ),
    ]

    response_schema = {
        'type': 'object',
        'properties': {
            'isFrontPageNews': {
                'type': 'boolean'
            }
        },
        'required': [
            'isFrontPageNews'
        ]
    }

    res = gemini.models.generate_content(
        model='gemini-2.5-flash',
        config=genai.types.GenerateContentConfig(
            temperature=0,
            response_mime_type='application/json',
            response_schema=response_schema,
        ),
        contents=contents
    )

    return res


def make_prompt(items_xml):
    with open('prompts/transform.txt') as f:
        prompt = f.read()

    return prompt.format(items_xml=items_xml, date=get_date())


def estimate_cost(res):
    model_version = res.model_version
    if 'gemini-2.5-pro' in res.model_version:
        model_version = 'gemini-2.5-pro'
    elif 'gemini-2.5-flash' in model_version:
        model_version = 'gemini-2.5-flash'

    input_cost = res.usage_metadata.prompt_token_count * MODELS[model_version]['input_token_cost']

    if res.usage_metadata.candidates_token_count:
        output_cost = res.usage_metadata.candidates_token_count * MODELS[model_version]['output_token_cost']
    else:
        # candidates_token_count is sometimes None, unclear why
        logger.info('no candidates_token_count, unable to estimate output cost')
        output_cost = 0

    return input_cost + output_cost


def make_blocks(content):
    blocks = []

    rendered = mistletoe.markdown(content)
    soup = BeautifulSoup(rendered, 'lxml')

    h1_tag = soup.find('h1')
    blocks.append({
        'type': 'header',
        'text': {
            'type': 'plain_text',
            'text': h1_tag.get_text(),
        },
    })

    for h2_tag in soup.find_all('h2'):
        blocks.extend([
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": h2_tag.get_text(),
                }
            },
            {
                "type": "divider",
            },
        ])

        ul_tag = h2_tag.find_next_sibling('ul')
        for li_tag in ul_tag.find_all('li'):
            mrkdwn_parts = ['◆ ']

            for content in li_tag.contents:
                if content.name == 'strong':
                    mrkdwn_parts.append(f'*{content.get_text()}*')
                elif content.name == 'a':
                    href = content.get('href')
                    text = content.get_text()
                    mrkdwn_parts.append(f'<{href}|{text}>')
                else:
                    mrkdwn_parts.append(content.get_text())

            mrkdwn = ''.join(mrkdwn_parts)

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": mrkdwn,
                }
            })

    return blocks


def run():
    items_xml = ''
    item_cache_path = Path('items.xml')

    if READ_ITEM_CACHE and item_cache_path.exists():
        logger.info('found cached news items')
        items_xml = item_cache_path.read_text()

    if not items_xml:
        sources = {
            'AP': {
                'crawler': get_ap_items,
            },
            'NHK': {
                'crawler': get_nhk_items,
            },
        }

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for source, details in sources.items():
                future = executor.submit(details['crawler'])
                futures[future] = source

            for future in as_completed(futures):
                source = futures[future]

                try:
                    sources[source]['items'] = future.result()
                except:
                    logger.exception(f'failed to get items for {source}')
                    continue

        all_items = []
        for details in sources.values():
            source_items = details.get('items')
            if source_items:
                all_items.extend(list(source_items.values()))

        random.shuffle(all_items)

        logger.info(f'testing {len(all_items)} news items')
        test_timer = Timer()
        test_progress = ProgressMeter(len(all_items), msg='tested {done}/{total} items ({percent}%)')
        front_page_items = []
        test_cost = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(test_item, item): item for item in all_items}

            for future in as_completed(futures):
                item = futures[future]

                try:
                    res = future.result()
                except:
                    logger.exception(f'failed to test item: {item["url"]}')
                    continue
                finally:
                    test_progress.increment()

                test_cost += estimate_cost(res)

                parsed_res = json.loads(res.text)
                if parsed_res['isFrontPageNews']:
                    front_page_items.append(item)
                else:
                    logger.info(f'ignoring item: {item["url"]}')

        test_timer.done()
        logger.info(
            f'done testing in {round(test_timer.latency, 2)}s (cost: ${round(test_cost, 4)}), '
            f'left with {len(front_page_items)} front-page items'
        )

        for item in front_page_items:
            items_xml += ITEM_XML_TEMPLATE.format(
                title=item['title'],
                url=item['url'],
                content=item['content']
            )

        if WRITE_ITEM_CACHE:
            item_cache_path.write_text(items_xml)

    if not items_xml:
        logger.info('aborting, no news items')
        return

    contents = [
        genai.types.Content(
            role='user',
            parts=[
                genai.types.Part.from_text(text=make_prompt(items_xml)),
            ]
        ),
    ]

    tokens_res = gemini.models.count_tokens(model='gemini-2.5-pro', contents=contents)
    logger.info(f'summarizing {tokens_res.total_tokens} tokens')

    generation_timer = Timer()
    # TODO: retries?
    res = gemini.models.generate_content(
        model='gemini-2.5-pro',
        config=genai.types.GenerateContentConfig(
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

    logger.info('pinging slack')
    blocks = make_blocks(res.text)
    slack.chat_postMessage(
        channel=SLACK_CHANNEL_ID,
        blocks=blocks,
        # fallback text for notifications only
        text=f'News for {get_date()}'
    )


def exception_handler(exception, event, context):
    logger.error('unhandled exception:', exc_info=exception)

    # Tells Zappa not to re-raise the exception, which in turn prevents Lambda
    # from retrying invocation.
    # https://github.com/zappa/Zappa/blob/0.60.1/zappa/handler.py#L252-L255
    return True
