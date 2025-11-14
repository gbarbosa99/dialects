import requests
from bs4 import BeautifulSoup as bf
import os
import time
import json

from urllib.parse import urljoin
from pydub import AudioSegment
import shutil

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
    os.makedirs(output_dir, exist_ok=True)
    updated_speakers = []

    speaker_urls = get_speaker_urls()

    for entry in speaker_urls:
        continent = entry['continent'].lower().replace(' ', '_')
        country = entry['country'].lower().replace(' ', '_')
        speaker = entry['speaker'].lower().replace(' ', '_')
        speaker_url = entry['url']

        base_filename = f"{continent}_{country}_{speaker}"
        audio_path = os.path.join(output_dir, base_filename + ".mp3")
        text_path = os.path.join(output_dir, base_filename + ".txt")

        try:
            page_resp = requests.get(speaker_url, headers=headers, timeout=15)
            page_resp.raise_for_status()

            soup = bf(page_resp.text, 'html.parser')
            audio_url = extract_audio_url(soup)

            if not audio_url:
                print(f"[WARN] No audio found for: {speaker}")
                continue

            # Make absolute URL
            audio_url = urljoin(speaker_url, audio_url)

            # Download audio with basic validation
            audio_resp = requests.get(audio_url, headers=headers, stream=True, timeout=30)
            audio_resp.raise_for_status()

            content_type = audio_resp.headers.get("Content-Type", "")
            if "audio" not in content_type and "mpeg" not in content_type:
                print(f"[WARN] Non-audio content for {speaker}: {content_type}")
                continue

            # Stream to disk
            with open(audio_path, 'wb') as f:
                for chunk in audio_resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Validate audio with pydub/ffmpeg
            try:
                audio = AudioSegment.from_file(audio_path)
            except Exception as e:
                print(f"[ERROR] Invalid audio for {speaker} ({audio_path}): {e}")
                os.remove(audio_path)
                continue

            # Optional: normalize to WAV, 16k mono
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            wav_path = audio_path.replace(".mp3", ".wav")
            audio.export(wav_path, format='wav')
            os.remove(audio_path)
            audio_path = wav_path

            # Get transcript text as you already do
            article = soup.find('div', class_='article')
            transcript = ""
            if article:
                paragraphs = article.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text.split()) > 50:
                        transcript = text
                        break  # first long paragraph is enough

            with open(text_path, 'w') as t:
                t.write(transcript)

            entry['audio_url'] = audio_url
            entry['local_audio_path'] = audio_path
            entry['transcript'] = transcript

            updated_speakers.append(entry)
            time.sleep(1)  # be polite to the site

        except Exception as e:
            print(f'[ERROR] Processing {speaker_url}: {e}')
            continue

    with open(os.path.join(output_dir, 'dialects_metadata.json'), 'w') as f:
        json.dump(updated_speakers, f, indent=2)

    return updated_speakers


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'raw')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    get_audio(OUTPUT_DIR)