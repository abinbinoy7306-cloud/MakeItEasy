import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables from .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL or SUPABASE_KEY is missing from .env"
    )

# Create Supabase client
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)