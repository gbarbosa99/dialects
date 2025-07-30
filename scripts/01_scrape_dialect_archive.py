import requests
from bs4 import BeautifulSoup as bf
import os
import time
import json

headers = {
"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0"
}

def extract_audio_url(soup):
    audio_tag = soup.find('audio')
    if audio_tag and audio_tag.get('src'):
        return audio_tag['src']

    if audio_tag:
        source_tag = audio_tag.find('source')
        if source_tag and source_tag.get('src'):
            return source_tag['src']

    if audio_tag:
        a_tag = audio_tag.find('a')
        if a_tag and a_tag.get('href'):
            return a_tag['href']
        
    all_links = soup.find_all('a', href=True)
    for link in all_links:
        if link['href'].endswith('.mp3'):
            return link['href']
    
    return None


def get_continent_urls():
    '''
    Iterate through the Dialect Archive dialects-accents page containing the dialect
    countries and regions. These are the first layer in the HTML crawl.
    '''

    url = "https://www.dialectsarchive.com/dialects-accents"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f'Failed to load page. Status code: {response.status_code}')
        return {}

    soup = bf(response.text, 'html.parser')

    # Dictionary to hold the links and names
    continent_urls = {}

    # Locate the main container
    article_div = soup.find('div', class_='article')
    if not article_div:
        print(f"Could not find the <div class='article'> section for ")
        return {}
    
    # Extract each <a> link and store its title and URL
    for link in article_div.find_all('a'):
        name = link.text.strip()
        href = link.get('href')
        continent_urls[name] = href

    return continent_urls


def get_country_urls():
    '''
    This function iterates through each continent and region, extracting each country. 
    Countries are the second layer in the HTML crawl.
    '''

    urls = []

    continent_urls = get_continent_urls()

    for continent, url in list(continent_urls.items()):
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f'Failed to load page. Status code: {response.status_code}')
            return {}

        soup = bf(response.text, 'html.parser')

        # Locate the main container
        article_div = soup.find('div', class_='article')
        if not article_div:
            print(f"Could not find the <div class='article'> section")
            continue

        # Extract each <a> link and store its title and URL
        for link in article_div.find_all('a'):
            country = link.text.strip()
            href = link.get('href')
            urls.append({
                "continent": continent,
                "country": country,
                "url": href
            })

    return urls

def get_speaker_urls():
    speakers = []

    country_urls = get_country_urls()

    for entry in country_urls:
        continent = entry["continent"]
        country = entry["country"]
        country_url = entry["url"]

        response = requests.get(country_url, headers=headers)
        soup = bf(response.text, 'html.parser')

        article_div = soup.find('div', class_='article')
        if not article_div:
            print(f"Could not find the <div class='article'> section")
            continue
        
        for link in article_div.find_all('a'):
            speaker = link.text.strip()
            href = link.get('href')
            speakers.append({
                "continent": continent,
                "country": country,
                "speaker": speaker,
                "url": href
            })
        
    return speakers

def get_audio(output_dir):
    """
    Use scraper to download audio from link <class="wp-audio-shortcode"> in src
    save audio to ../data/raw
    save transcriptions to a json file in ../data/raw/transcription.json with audio file names

    input:
    flat list - [continent, country, speaker_id]

    output:
    mp4 files
    transcription.json
    """
    os.makedirs(output_dir, exist_ok=True)
    updated_speakers = []

    speaker_urls = get_speaker_urls()

    for entry in speaker_urls:
        continent = entry['continent'].lower().replace(' ', '_')
        country = entry['country'].lower().replace(' ', '_')
        speaker = entry['speaker'].lower().replace(' ', '_')
        speaker_url = entry['url']

        base_filename = f"{continent}_{speaker}"
        audio_path = os.path.join(output_dir, base_filename + ".mp3")
        text_path = os.path.join(output_dir, base_filename + ".txt")

        try:
            response = requests.get(speaker_url, headers=headers)

            if response.status_code != 200:
                print(f'Failed to load page. Status code: {response.status_code}')
                return {}
            
            soup = bf(response.text, 'html.parser')

            # Locate the main container
            audio_url = extract_audio_url(soup)

            if not audio_url:
                print(f"No audio found for: {speaker}")
                continue

            audio_data = requests.get(audio_url).content
            with open(audio_path, 'wb') as f:
                f.write(audio_data)

            # Get transcript
            article = soup.find('div', class_='article')
            transcript = ""
            if article:
                paragraphs = article.find_all('p')
                for p in paragraphs:
                    text=p.get_text(strip=True)
                    length = len(text.split())
                    if length > 50:
                        transcript = text

            with open(text_path, 'w') as t:
                t.write(transcript)
            
            entry['audio_url'] = audio_url
            entry['local_audio_path'] = audio_path
            entry['transcript'] = transcript

            updated_speakers.append(entry)
            time.sleep(1)

        except Exception as e:
            print(f'Error processing {speaker_url}: {e}')
            continue

    with open(os.path.join(output_dir, 'dialects_metadata.json'), 'w') as f:
        json.dump(updated_speakers, f, indent=2)

    return updated_speakers


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'raw')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    get_audio(OUTPUT_DIR)