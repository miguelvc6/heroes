import requests
import time
import json
from urllib.parse import quote
from tqdm import tqdm
from bs4 import BeautifulSoup

BASE_URL = "https://hero.fandom.com/"
API_URL = BASE_URL + "api.php"


def get_all_page_ids(num_pages=None):
    all_pages = []
    apfrom = None
    count = 0
    while num_pages is None or num_pages > 0:
        if num_pages is not None:
            num_pages -= 1

        params = {"action": "query", "list": "allpages", "aplimit": "500", "format": "json"}
        if apfrom:
            params["apfrom"] = apfrom

        response = requests.get(API_URL, params=params)
        data = response.json()
        pages = data["query"]["allpages"]
        if not pages:
            break
        for page in pages:
            page["url"] = f"{BASE_URL}wiki/{quote(page['title'])}"
        all_pages.extend(pages)

        if "continue" in data:
            apfrom = data["continue"]["apcontinue"]
        else:
            break
        count = len(all_pages)
        print(f"Count: {count}")

        time.sleep(1)

    return all_pages


def extract_character_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    character = {"type": None, "text_content": "", "friends": [], "neutral": [], "enemies": []}
    content = soup.find("div", id="content")
    if content:
        content = content.find("div", class_="mw-parser-output")
        if content:
            aside = content.find("aside")
            if aside and aside.h2 and aside.h2.div and aside.h2.div.span:
                character["type"] = aside.h2.div.span.text

            character["text_content"] = "".join([p.get_text() for p in content.find_all("p", recursive=False)[1:]])

            for relation in ["Friends.2FAcquaintances", "Neutral", "Enemies"]:
                relation_span = content.find("span", id=relation)
                if relation_span and relation_span.parent:
                    relation_ul = relation_span.parent.find_next_sibling("ul")
                    if relation_ul:
                        character[relation.lower()] = [
                            BASE_URL[:-1] + li.a["href"]
                            for li in relation_ul.find_all("li")
                            if li.a and "href" in li.a.attrs
                        ]

    return character


def main():
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    print("Getting all page ids...")
    all_page_ids = get_all_page_ids()
    print(f"Total pages retrieved: {len(all_page_ids)}")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}\n")

    with open("data/all_pages.jsonl", "w") as file:
        for page in all_page_ids:
            json.dump({"pageid": page["pageid"], "title": page["title"], "url": page["url"]}, file)
            file.write("\n")

    print("Extracting character info...")
    characters = []
    for page in tqdm(all_page_ids):
        url = page["url"]
        character = page.copy()
        max_retries = 16
        retry_delay = 0.5
        for attempt in range(max_retries):
            try:
                character.update(extract_character_info(url))
                break
            except requests.RequestException:
                # Exponential backoff
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to extract info for {url} after {max_retries} attempts")
                    character = None
        if character:
            characters.append(character)
        time.sleep(0.5)

    print(f"Total characters processed: {len(characters)}")

    with open("data/characters.jsonl", "w", encoding="utf-8") as f:
        for character in characters:
            json.dump(character, f, ensure_ascii=False)
            f.write("\n")

    print("Characters saved to characters.jsonl")


if __name__ == "__main__":
    main()
