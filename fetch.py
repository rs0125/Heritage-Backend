import wikipedia
import os

# 1. Define the folder to save your documents
# This MUST match the folder your 'ingest.py' script reads from
DOCS_DIR = "./your-docs-folder"

# 2. Create the directory if it doesn't exist
os.makedirs(DOCS_DIR, exist_ok=True)

# 3. Define the Wikipedia topics you want to ingest
TOPICS = [
    "History of India",
    "Mughal Empire",
    "Maurya Empire",
    "Gupta Empire",
    "Maratha Empire",
    "Vijayanagara Empire",
    "Chola dynasty",
    "Indian independence movement",
    "Ashoka",
    "Akbar",
    "Timeline of Indian history",
]

print(f"Starting to fetch {len(TOPICS)} topics from Wikipedia...")

for topic in TOPICS:
    print(f"Fetching '{topic}'...")
    try:
        # Get the Wikipedia page
        # auto_suggest=False stops it from guessing if the title is slightly off
        page = wikipedia.page(topic, auto_suggest=False)
        content = page.content

        # Clean the topic name to create a valid filename
        filename = f"{topic.replace(' ', '_').lower()}.txt"
        filepath = os.path.join(DOCS_DIR, filename)

        # Save the content as a .txt file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  > Saved '{topic}' to {filepath}")

    except wikipedia.exceptions.DisambiguationError as e:
        print(
            f"  > Skipped '{topic}': Disambiguation page. Try a more specific topic from its options."
        )
    except wikipedia.exceptions.PageError:
        print(f"  > Skipped '{topic}': Page not found.")
    except Exception as e:
        print(f"  > Skipped '{topic}': An error occurred: {e}")

print("\nData fetching complete.")
