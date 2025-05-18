import requests
import time
import os
from tqdm import tqdm

# - Configuration -
START_ID = 1
END_ID = 100 # Download books up to this ID (inclusive)
BASE_URL = "https://www.gutenberg.org/ebooks/{id}.txt.utf-8"
OUTPUT_FILE = "corpus.txt"
# Delay between requests in seconds
REQUEST_DELAY = 0.5
# Custom User-Agent
HEADERS = {
    'User-Agent': 'MyGutenbergCorpusBuilder/1.0 (Python script for personal use)'
}

# Ensure the output directory exists
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

print(f"Starting download process for IDs {START_ID} to {END_ID}.")
print(f"Output will be saved to: {OUTPUT_FILE}")
print(f"Delay between requests: {REQUEST_DELAY} seconds")

success_count = 0
fail_count = 0
total_ids = END_ID - START_ID + 1

# Open the output file in append mode with UTF-8 encoding
with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
    # Use tqdm for a progress bar
    for book_id in tqdm(range(START_ID, END_ID + 1), desc="Downloading Books", total=total_ids):
        url = BASE_URL.format(id=book_id)

        try:
            # Make the HTTP GET request
            response = requests.get(url, headers=HEADERS, timeout=20) 

            # Check if the request was successful (HTTP status code 200)
            response.raise_for_status() 

            # Decode the text content
            book_text = response.text

            # Write a separator and the book text to the output file
            outfile.write(f"\n\n--- START BOOK ID: {book_id} ---\n\n")
            outfile.write(book_text)
            outfile.write(f"\n\n--- END BOOK ID: {book_id} ---\n\n")

            success_count += 1

        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors (like 404 Not Found) gracefully
            fail_count += 1
        except requests.exceptions.RequestException as e:
            # Handle other network/request errors (timeout, connection error, etc.)
            print(f"ID {book_id}: Failed - Request Error: {e}")
            fail_count += 1
        except Exception as e:
            # Catch any other unexpected errors during processing
            print(f"ID {book_id}: Failed - Unexpected Error: {e}")
            fail_count += 1

        # Pause between requests
        time.sleep(REQUEST_DELAY)

print("\n-----------------------------------------")
print("Download process finished.")
print(f"Successfully downloaded and appended: {success_count}")
print(f"Failed attempts (e.g., not found, errors): {fail_count}")
print(f"Total IDs checked: {total_ids}")
print(f"Corpus saved to: {OUTPUT_FILE}")
print("-----------------------------------------")