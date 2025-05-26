import logging
from datetime import datetime
from logging.config import dictConfig

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

gemini = genai.Client(
    http_options=genai.types.HttpOptions(timeout=120 * 1000)
)


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


def run():
    session = requests.Session()

    url = 'https://news.google.com/rss/search?q=site:apnews.com+when:1d'
    response = session.get(url)

    soup = BeautifulSoup(response.text, 'lxml-xml')
    items = soup.find_all('item')

    print(f'got {len(items)} news items')

    news = []
    for item in items:
        news.append(f'<item>{item.title}{item.link}</item>')

    joined_news = '\n'.join(news)

    contents = [
        genai.types.Content(
            role='user',
            parts=[
                genai.types.Part.from_text(text=joined_news),
            ]
        ),
    ]

    res = gemini.models.generate_content(
        model='gemini-2.5-flash-preview-05-20',
        # model='gemini-2.5-pro-preview-05-06',
        config=genai.types.GenerateContentConfig(
            system_instruction=make_prompt(),
            temperature=0,
        ),
        contents=contents
    )

    print(res.text)

    cost = estimate_cost(res)
    print(f'estimated cost: {round(cost, 4)}')
