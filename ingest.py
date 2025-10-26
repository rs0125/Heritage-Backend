import os
import time  # <-- Import the time module
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

# --- !! SET THESE !! ---
# Get this from Supabase: Project Settings > Database > Connection Pooling (Session)
CONNECTION_STRING = "xxx"

# Get from OpenAI
os.environ["OPENAI_API_KEY"] = (
    "xxx"
)
# ------------------------

# 1. Load your documents
loader = DirectoryLoader("./your-docs-folder/", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# 2. Split them into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")

# 3. Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# --- NEW BATCHED INGESTION LOGIC ---

# 4. Initialize the vector store connection
# We tell it to wipe the collection, but don't give it docs yet.
print("Initializing vector store and clearing old collection...")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="documents",
    connection=CONNECTION_STRING,
    pre_delete_collection=True,  # This will wipe the 'documents' table
)

# 5. Define batch size and create batches
BATCH_SIZE = 100  # Process 100 chunks at a time.
batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
total_batches = len(batches)
print(f"Created {total_batches} batches of size {BATCH_SIZE}.")

# 6. Loop through batches with a delay
for i, batch in enumerate(batches):
    print(f"Ingesting batch {i + 1}/{total_batches} ({len(batch)} documents)...")
    try:
        # Add the documents in the current batch
        vector_store.add_documents(batch)

        if i < total_batches - 1:  # Don't sleep after the last batch
            print(f"Batch {i + 1} complete. Waiting 60 seconds to avoid rate limit...")
            time.sleep(60)  # Wait for 1 minute

    except Exception as e:
        print(f"Error ingesting batch {i + 1}: {e}")
        print("Skipping this batch and continuing after a 60-second wait...")
        if i < total_batches - 1:
            time.sleep(60)

print("\n--- Ingestion complete. ---")
