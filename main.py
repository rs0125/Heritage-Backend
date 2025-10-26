import os
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import bcrypt
from typing import List

# --- NEW IMPORTS FOR FIX & LOGGING ---
import traceback  # For detailed error logging
import logging  # For verbose logging
from operator import itemgetter  # The main fix for your chain

# --- .env Imports ---
from dotenv import load_dotenv

# --- DB Imports ---
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    select,
    func,
)
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.exc import IntegrityError

# --- LangChain Imports ---
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.prompts import ChatPromptTemplate

# --- 1. Load Environment Variables & Setup Logger ---
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- !! SET THESE IN YOUR .env FILE !! ---
CONNECTION_STRING = os.getenv("SUPABASE_CONNECTION_STRING")
if not CONNECTION_STRING or not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "Missing environment variables. Please check your .env file."
    )
# ----------------------------------------


# --- 2. SQLAlchemy Database Setup ---
engine = create_engine(CONNECTION_STRING)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    is_user_message = Column(Boolean, nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- 3. Auth Setup (Using your bcrypt functions) ---
security = HTTPBasic()


def get_password_hash(password: str) -> str:
    """Hash password using bcrypt with 72-byte truncation"""
    password_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password using bcrypt with 72-byte truncation"""
    password_bytes = plain_password.encode("utf-8")[:72]
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def get_current_user(
    credentials: HTTPBasicCredentials = Depends(security), db: Session = Depends(get_db)
):
    try:
        user = db.scalars(
            select(User).where(User.username == credentials.username)
        ).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )

        if not verify_password(credentials.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )
        return {"username": user.username, "id": user.id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}",
            headers={"WWW-Authenticate": "Basic"},
        )


# --- 4. FastAPI App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://172.16.180.115:3000",
        "http://192.168.240.41:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 5. RAG Setup ---
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = PGVector(
    connection=CONNECTION_STRING,
    collection_name="documents",
    embeddings=embeddings,
)
retriever = vector_store.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
template = """
You are an AI assistant. Answer the user's question based on the
following context. If you don't know the answer, just say so.
Don't try to make up an answer.

Context:
{context}

Chat History:
{history}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


#
# --- THIS IS THE FIXED RAG CHAIN ---
#
# We use itemgetter to properly route the "question" and "history"
# from the input dictionary to the correct parts of the chain.
#
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --- 6. Pydantic Models ---
class UserCreate(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str
    history: List[dict]


# --- 7. API Endpoints ---
@app.post("/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        hashed_password = get_password_hash(user.password)
        db_user = User(username=user.username, hashed_password=hashed_password)

        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return {"message": "User created successfully", "user_id": db_user.id}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username already registered")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


#
# --- THIS IS THE NEW "VERBOSE" CHAT ENDPOINT ---
#
@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Send a chat message and get AI response"""
    user_id = current_user["id"]
    logger.info(f"Chat request for user {user_id} with message: {request.message}")

    # Log user message
    try:
        user_log = ChatLog(
            user_id=user_id, is_user_message=True, message=request.message
        )
        db.add(user_log)
        db.commit()
        logger.info("Successfully logged user message to DB.")
    except Exception as e:
        logger.error(f"Failed to log user message: {e}")
        db.rollback()
        # We can still proceed, but this is good to know

    # Format history
    formatted_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in request.history]
    )

    try:
        # --- VERBOSE DEBUGGING PART ---
        logger.info("Invoking RAG chain...")

        # Prepare the input for the chain
        chain_input = {"question": request.message, "history": formatted_history}
        logger.info(f"Chain input: {chain_input}")

        # Invoke the chain
        response_text = rag_chain.invoke(chain_input)

        logger.info(f"RAG chain successful. Response: {response_text[:50]}...")
        # --- END OF VERBOSE DEBUGGING PART ---

    except Exception as e:
        # THIS WILL PRINT THE *EXACT* ERROR TO YOUR TERMINAL
        logger.error(f"--- RAG CHAIN FAILED ---")
        logger.error(f"Error Type: {type(e)}")
        logger.error(f"Error Details: {str(e)}")
        logger.error(f"Traceback: \n{traceback.format_exc()}")
        logger.error(f"--- END OF TRACEBACK ---")

        db.rollback()  # Rollback the user message log if chain fails
        raise HTTPException(
            status_code=500,
            detail=f"Error in RAG chain. Check backend logs for details. Error: {str(e)}",
        )

    # Log bot response
    try:
        bot_log = ChatLog(user_id=user_id, is_user_message=False, message=response_text)
        db.add(bot_log)
        db.commit()
        logger.info("Successfully logged bot response to DB.")
    except Exception as e:
        logger.error(f"Failed to log bot response: {e}")
        db.rollback()

    return {"response": response_text}


@app.get("/history")
async def get_history(
    current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get chat history for the current user"""
    user_id = current_user["id"]

    logs = db.scalars(
        select(ChatLog)
        .where(ChatLog.user_id == user_id)
        .order_by(ChatLog.created_at.asc())
    ).all()

    history = []
    for row in logs:
        history.append(
            {
                "role": "user" if row.is_user_message else "assistant",
                "content": row.message,
            }
        )
    return history


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "RAG Chat API is running"}


# --- 8. Run the app ---
if __name__ == "__main__":
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    # Use uvicorn from the command line instead: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
