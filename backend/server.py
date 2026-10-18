from fastapi import FastAPI, APIRouter, HTTPException, Header, Request
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import json
import logging
import uuid
import re
import requests
import httpx
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta

from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
from emergentintegrations.payments.stripe.checkout import (
    StripeCheckout, CheckoutSessionRequest,
)
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')
STRIPE_API_KEY = os.environ.get('STRIPE_API_KEY', 'sk_test_emergent')
GOOGLE_WEB_CLIENT_ID = os.environ.get('GOOGLE_WEB_CLIENT_ID', '')
GOOGLE_ANDROID_CLIENT_ID = os.environ.get('GOOGLE_ANDROID_CLIENT_ID', '')
GOOGLE_IOS_CLIENT_ID = os.environ.get('GOOGLE_IOS_CLIENT_ID', '')
ACCEPTED_GOOGLE_AUDIENCES = [c for c in [GOOGLE_WEB_CLIENT_ID, GOOGLE_ANDROID_CLIENT_ID, GOOGLE_IOS_CLIENT_ID] if c]
GEMINI_MODEL = "gemini-3-flash-preview"

stripe_checkout = StripeCheckout(api_key=STRIPE_API_KEY)

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------- Models ----------
class PantryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    quantity: Optional[str] = None
    source: str = "manual"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PantryItemCreate(BaseModel):
    name: str
    quantity: Optional[str] = None
    source: str = "manual"


class PantryBulkAdd(BaseModel):
    items: List[PantryItemCreate]


class ScanImageRequest(BaseModel):
    image_base64: str


class Recipe(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    cuisine: str
    country: str
    cook_time_minutes: int
    difficulty: str
    description: str
    ingredients: List[str]
    missing_ingredients: List[str] = []
    instructions: List[str]
    match_percent: int = 0
    image_query: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RecipeGenerateRequest(BaseModel):
    ingredients: List[str]
    country: str = "US"
    count: int = 5


class Settings(BaseModel):
    country: str = "US"
    country_name: str = "United States"


class GoogleSessionRequest(BaseModel):
    session_id: str


class GoogleIdTokenRequest(BaseModel):
    id_token: str


class User(BaseModel):
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    pro_expires_at: Optional[str] = None  # ISO datetime
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CheckoutRequest(BaseModel):
    country: str = "IN"
    origin_url: str  # frontend origin for success/cancel


class ExpiryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    expiry_date: str  # YYYY-MM-DD
    quantity: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExpiryItemCreate(BaseModel):
    name: str
    expiry_date: str
    quantity: Optional[str] = None


class VoiceChefMessage(BaseModel):
    session_id: str
    message: str


# ---------- Helpers ----------
def _strip_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    candidates = [i for i in [text.find('{'), text.find('[')] if i != -1]
    first = min(candidates) if candidates else 0
    last = max(text.rfind('}'), text.rfind(']'))
    if last > first:
        text = text[first:last + 1]
    return text


async def _gemini_chat(session_id: str, system: str) -> LlmChat:
    return LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message=system,
    ).with_model("gemini", GEMINI_MODEL)


COUNTRY_CUISINE_MAP = {
    "IN": ("India", "Indian"), "US": ("United States", "American"), "GB": ("United Kingdom", "British"),
    "IT": ("Italy", "Italian"), "MX": ("Mexico", "Mexican"), "JP": ("Japan", "Japanese"),
    "CN": ("China", "Chinese"), "TH": ("Thailand", "Thai"), "FR": ("France", "French"),
    "ES": ("Spain", "Spanish"), "GR": ("Greece", "Greek"), "TR": ("Turkey", "Turkish"),
    "BR": ("Brazil", "Brazilian"), "KR": ("South Korea", "Korean"), "VN": ("Vietnam", "Vietnamese"),
    "DE": ("Germany", "German"), "LB": ("Lebanon", "Lebanese"), "MA": ("Morocco", "Moroccan"),
    "EG": ("Egypt", "Egyptian"), "PK": ("Pakistan", "Pakistani"), "BD": ("Bangladesh", "Bangladeshi"),
    "NG": ("Nigeria", "Nigerian"), "AU": ("Australia", "Australian"), "CA": ("Canada", "Canadian"),
}

# Pro plan localized pricing (server side authoritative)
PRICING_BY_COUNTRY: Dict[str, Dict] = {
    "IN": {"currency": "inr", "amount": 100.0, "symbol": "₹", "display": "₹100"},
    "US": {"currency": "usd", "amount": 1.20, "symbol": "$", "display": "$1.20"},
    "GB": {"currency": "gbp", "amount": 0.95, "symbol": "£", "display": "£0.95"},
    "EU": {"currency": "eur", "amount": 1.10, "symbol": "€", "display": "€1.10"},
    "FR": {"currency": "eur", "amount": 1.10, "symbol": "€", "display": "€1.10"},
    "DE": {"currency": "eur", "amount": 1.10, "symbol": "€", "display": "€1.10"},
    "IT": {"currency": "eur", "amount": 1.10, "symbol": "€", "display": "€1.10"},
    "ES": {"currency": "eur", "amount": 1.10, "symbol": "€", "display": "€1.10"},
    "GR": {"currency": "eur", "amount": 1.10, "symbol": "€", "display": "€1.10"},
    "JP": {"currency": "jpy", "amount": 180.0, "symbol": "¥", "display": "¥180"},
    "CN": {"currency": "cny", "amount": 8.50, "symbol": "¥", "display": "¥8.50"},
    "CA": {"currency": "cad", "amount": 1.65, "symbol": "C$", "display": "C$1.65"},
    "AU": {"currency": "aud", "amount": 1.80, "symbol": "A$", "display": "A$1.80"},
    "MX": {"currency": "mxn", "amount": 24.0, "symbol": "MX$", "display": "MX$24"},
    "BR": {"currency": "brl", "amount": 6.00, "symbol": "R$", "display": "R$6.00"},
    "KR": {"currency": "krw", "amount": 1600.0, "symbol": "₩", "display": "₩1,600"},
    "TH": {"currency": "thb", "amount": 42.0, "symbol": "฿", "display": "฿42"},
    "VN": {"currency": "vnd", "amount": 28000.0, "symbol": "₫", "display": "₫28,000"},
    "TR": {"currency": "try", "amount": 40.0, "symbol": "₺", "display": "₺40"},
    "PK": {"currency": "pkr", "amount": 330.0, "symbol": "₨", "display": "₨330"},
    "BD": {"currency": "bdt", "amount": 130.0, "symbol": "৳", "display": "৳130"},
    "NG": {"currency": "ngn", "amount": 1800.0, "symbol": "₦", "display": "₦1,800"},
    "LB": {"currency": "usd", "amount": 1.20, "symbol": "$", "display": "$1.20"},
    "MA": {"currency": "mad", "amount": 12.0, "symbol": "MAD", "display": "MAD 12"},
    "EG": {"currency": "egp", "amount": 60.0, "symbol": "E£", "display": "E£60"},
}


def pricing_for(country: str) -> Dict:
    return PRICING_BY_COUNTRY.get((country or "IN").upper(), PRICING_BY_COUNTRY["IN"])


def country_to_cuisine(code: str):
    code = (code or "US").upper()
    return COUNTRY_CUISINE_MAP.get(code, (code, code))


async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[User]:
    """Bearer token auth helper. Returns User or None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    session = await db.user_sessions.find_one({"session_token": token}, {"_id": 0})
    if not session:
        return None
    # Check expiry
    exp = session.get("expires_at")
    if exp:
        exp_dt = datetime.fromisoformat(exp) if isinstance(exp, str) else exp
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        if exp_dt < datetime.now(timezone.utc):
            return None
    user = await db.users.find_one({"user_id": session["user_id"]}, {"_id": 0})
    if not user:
        return None
    return User(**user)


async def require_user(authorization: Optional[str] = Header(None)) -> User:
    user = await get_current_user(authorization)
    if not user:
        raise HTTPException(401, "Not authenticated")
    return user


def is_pro(user: User) -> bool:
    if not user.pro_expires_at:
        return False
    exp = datetime.fromisoformat(user.pro_expires_at)
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp > datetime.now(timezone.utc)


async def require_pro(user: User = None, authorization: Optional[str] = Header(None)) -> User:
    if user is None:
        user = await require_user(authorization)
    if not is_pro(user):
        raise HTTPException(402, "Pro subscription required")
    return user


# ---------- Routes ----------
@api_router.get("/")
async def root():
    return {"message": "Fridgy API", "status": "ok"}


# Settings
@api_router.get("/settings", response_model=Settings)
async def get_settings():
    doc = await db.settings.find_one({"_id": "user"}, {"_id": 0})
    if not doc:
        return Settings()
    return Settings(**doc)


@api_router.put("/settings", response_model=Settings)
async def update_settings(s: Settings):
    country_name, _ = country_to_cuisine(s.country)
    s.country_name = country_name
    await db.settings.update_one({"_id": "user"}, {"$set": s.dict()}, upsert=True)
    return s


# Pantry
@api_router.get("/pantry", response_model=List[PantryItem])
async def list_pantry():
    items = await db.pantry.find({}, {"_id": 0}).sort("created_at", -1).to_list(500)
    return [PantryItem(**i) for i in items]


@api_router.post("/pantry", response_model=PantryItem)
async def add_pantry(item: PantryItemCreate):
    obj = PantryItem(**item.dict())
    await db.pantry.insert_one(obj.dict())
    return obj


@api_router.post("/pantry/bulk", response_model=List[PantryItem])
async def add_pantry_bulk(payload: PantryBulkAdd):
    objs = [PantryItem(**i.dict()) for i in payload.items]
    if objs:
        await db.pantry.insert_many([o.dict() for o in objs])
    return objs


@api_router.delete("/pantry/{item_id}")
async def delete_pantry(item_id: str):
    res = await db.pantry.delete_one({"id": item_id})
    return {"deleted": res.deleted_count}


@api_router.delete("/pantry")
async def clear_pantry():
    res = await db.pantry.delete_many({})
    return {"deleted": res.deleted_count}


# Scanning
@api_router.post("/scan/fridge")
async def scan_fridge(req: ScanImageRequest):
    if not EMERGENT_LLM_KEY:
        raise HTTPException(500, "LLM key not configured")
    try:
        chat = await _gemini_chat(
            session_id=f"fridge-{uuid.uuid4()}",
            system="You identify food ingredients visible in a refrigerator. Return ONLY valid JSON.",
        )
        msg = UserMessage(
            text=(
                "Identify every distinct food ingredient visible in this fridge photo. "
                'Return strict JSON: {"ingredients":[{"name":"...","quantity":"..."}]} '
                "Use simple lowercase names. If nothing visible, return empty array."
            ),
            file_contents=[ImageContent(image_base64=req.image_base64)],
        )
        resp = await chat.send_message(msg)
        data = json.loads(_strip_json(str(resp)))
        return {"ingredients": data.get("ingredients", []) if isinstance(data, dict) else []}
    except Exception as e:
        logger.error(f"scan_fridge error: {e}")
        raise HTTPException(500, f"Scan failed: {str(e)}")


@api_router.post("/scan/receipt")
async def scan_receipt(req: ScanImageRequest):
    if not EMERGENT_LLM_KEY:
        raise HTTPException(500, "LLM key not configured")
    try:
        chat = await _gemini_chat(
            session_id=f"receipt-{uuid.uuid4()}",
            system="You extract grocery items from a receipt. Return ONLY valid JSON.",
        )
        msg = UserMessage(
            text=(
                "Extract grocery food items from this receipt. Skip totals/taxes/store names. "
                'Return strict JSON: {"items":[{"name":"...","quantity":"..."}]}'
            ),
            file_contents=[ImageContent(image_base64=req.image_base64)],
        )
        resp = await chat.send_message(msg)
        data = json.loads(_strip_json(str(resp)))
        return {"items": data.get("items", []) if isinstance(data, dict) else []}
    except Exception as e:
        logger.error(f"scan_receipt error: {e}")
        raise HTTPException(500, f"Scan failed: {str(e)}")


@api_router.get("/scan/barcode/{barcode}")
async def scan_barcode(barcode: str):
    try:
        url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
        r = requests.get(url, timeout=10, headers={"User-Agent": "FridgeChef/1.0"})
        if r.status_code != 200:
            raise HTTPException(404, "Product not found")
        data = r.json()
        product = data.get("product")
        if not product or data.get("status") == 0:
            return {"found": False, "barcode": barcode}
        return {
            "found": True,
            "barcode": barcode,
            "name": product.get("product_name") or product.get("generic_name") or "Unknown",
            "brand": product.get("brands", ""),
            "categories": product.get("categories", ""),
            "image_url": product.get("image_front_small_url") or product.get("image_url", ""),
            "quantity": product.get("quantity", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"barcode error: {e}")
        raise HTTPException(500, f"Lookup failed: {str(e)}")


# Recipes
@api_router.post("/recipes/generate", response_model=List[Recipe])
async def generate_recipes(req: RecipeGenerateRequest):
    if not req.ingredients:
        raise HTTPException(400, "No ingredients provided")
    country_name, cuisine = country_to_cuisine(req.country)
    try:
        chat = await _gemini_chat(
            session_id=f"recipes-{uuid.uuid4()}",
            system=(
                f"You are an authentic {cuisine} chef from {country_name}. "
                f"You ONLY suggest traditional/popular {cuisine} recipes from {country_name}. "
                "Never suggest cuisines from other countries. Return ONLY valid JSON."
            ),
        )
        prompt = (
            f"Available ingredients: {', '.join(req.ingredients)}.\n"
            f"Generate {req.count} authentic {cuisine} recipes (popular in {country_name}) "
            "primarily using available ingredients. Pantry staples assumed available.\n"
            "Return strict JSON array of recipes with: title, cuisine, country, "
            "cook_time_minutes, difficulty (easy/medium/hard), description, "
            "ingredients (list with quantities), missing_ingredients (list), instructions (list), "
            "match_percent (0-100), image_query."
        )
        resp = await chat.send_message(UserMessage(text=prompt))
        data = json.loads(_strip_json(str(resp)))
        if not isinstance(data, list):
            data = data.get("recipes", []) if isinstance(data, dict) else []
        recipes = []
        for r in data:
            try:
                recipes.append(Recipe(**r))
            except Exception as ex:
                logger.warning(f"skip invalid recipe: {ex}")
        if recipes:
            await db.recipes.insert_many([r.dict() for r in recipes])
        return recipes
    except Exception as e:
        logger.error(f"generate_recipes error: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")


@api_router.get("/recipes", response_model=List[Recipe])
async def list_recipes():
    items = await db.recipes.find({}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return [Recipe(**i) for i in items]


@api_router.get("/recipes/{recipe_id}", response_model=Recipe)
async def get_recipe(recipe_id: str):
    item = await db.recipes.find_one({"id": recipe_id}, {"_id": 0})
    if not item:
        raise HTTPException(404, "Recipe not found")
    return Recipe(**item)


@api_router.delete("/recipes")
async def clear_recipes():
    res = await db.recipes.delete_many({})
    return {"deleted": res.deleted_count}


# ---------- AUTH (Direct Google OAuth) ----------
@api_router.post("/auth/google/verify")
async def google_verify(req: GoogleIdTokenRequest):
    """Verify a Google ID token directly (NOT via Emergent). Returns app session_token + user."""
    if not ACCEPTED_GOOGLE_AUDIENCES:
        raise HTTPException(500, "Google client id not configured")
    try:
        # Decode header to find audience, then verify with that specific client_id
        info = None
        last_err = None
        for client_id in ACCEPTED_GOOGLE_AUDIENCES:
            try:
                info = google_id_token.verify_oauth2_token(
                    req.id_token, google_requests.Request(), client_id,
                )
                break
            except Exception as e:
                last_err = e
                continue
        if info is None:
            raise last_err or ValueError("No matching audience")
        if info.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
            raise ValueError("Bad issuer")
    except Exception as e:
        logger.warning(f"Google token verify failed: {e}")
        raise HTTPException(401, "Invalid Google ID token")

    email = info.get("email")
    if not email:
        raise HTTPException(401, "Email scope missing")

    # Upsert user
    existing = await db.users.find_one({"email": email}, {"_id": 0})
    if existing:
        user_id = existing["user_id"]
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {
                "name": info.get("name", existing.get("name") or email),
                "picture": info.get("picture", existing.get("picture")),
            }},
        )
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        await db.users.insert_one({
            "user_id": user_id,
            "email": email,
            "name": info.get("name", email),
            "picture": info.get("picture"),
            "pro_expires_at": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    session_token = uuid.uuid4().hex + uuid.uuid4().hex  # 64 chars
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.update_one(
        {"session_token": session_token},
        {"$set": {
            "session_token": session_token,
            "user_id": user_id,
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }},
        upsert=True,
    )

    user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    return {
        "session_token": session_token,
        "user": user,
        "is_pro": is_pro(User(**user)),
    }


# Legacy Emergent-managed endpoint kept for backward compat (no longer used)
@api_router.post("/auth/google/session")
async def google_session(req: GoogleSessionRequest):
    """Exchange Emergent session_id for app session_token + user record."""
    async with httpx.AsyncClient(timeout=15.0) as cli:
        r = await cli.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": req.session_id},
        )
        if r.status_code != 200:
            raise HTTPException(401, "Invalid session_id")
        data = r.json()
    email = data.get("email")
    if not email:
        raise HTTPException(401, "No email in session data")
    session_token = data["session_token"]
    # Upsert user by email
    existing = await db.users.find_one({"email": email}, {"_id": 0})
    if existing:
        user_id = existing["user_id"]
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        await db.users.insert_one({
            "user_id": user_id,
            "email": email,
            "name": data.get("name", email),
            "picture": data.get("picture"),
            "pro_expires_at": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    # Store session
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.update_one(
        {"session_token": session_token},
        {"$set": {
            "session_token": session_token,
            "user_id": user_id,
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }},
        upsert=True,
    )
    user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    return {
        "session_token": session_token,
        "user": user,
        "is_pro": is_pro(User(**user)),
    }


@api_router.get("/auth/me")
async def auth_me(authorization: Optional[str] = Header(None)):
    user = await require_user(authorization)
    return {"user": user.dict(), "is_pro": is_pro(user)}


@api_router.post("/auth/logout")
async def auth_logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        await db.user_sessions.delete_one({"session_token": token})
    return {"ok": True}


# ---------- SUBSCRIPTION ----------
@api_router.get("/subscription/pricing")
async def subscription_pricing(country: str = "IN"):
    p = pricing_for(country)
    return {**p, "country": country.upper(), "interval": "month"}


@api_router.post("/subscription/checkout")
async def subscription_checkout(req: CheckoutRequest, authorization: Optional[str] = Header(None)):
    user = await require_user(authorization)
    p = pricing_for(req.country)
    origin = req.origin_url.rstrip('/')
    try:
        cr = CheckoutSessionRequest(
            amount=p["amount"],
            currency=p["currency"],
            success_url=f"{origin}/pro/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{origin}/pro",
            metadata={"user_id": user.user_id, "country": req.country.upper(), "plan": "pro_monthly"},
        )
        resp = await stripe_checkout.create_checkout_session(cr)
        # Record pending payment
        await db.payment_transactions.insert_one({
            "session_id": resp.session_id,
            "user_id": user.user_id,
            "country": req.country.upper(),
            "currency": p["currency"],
            "amount": p["amount"],
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        return {"url": resp.url, "session_id": resp.session_id, "pricing": p}
    except Exception as e:
        logger.error(f"checkout error: {e}")
        raise HTTPException(500, f"Checkout failed: {str(e)}")


@api_router.get("/subscription/status/{session_id}")
async def subscription_status(session_id: str, authorization: Optional[str] = Header(None)):
    user = await require_user(authorization)
    try:
        status = await stripe_checkout.get_checkout_status(session_id)
        # Check if already activated (idempotent)
        tx = await db.payment_transactions.find_one({"session_id": session_id}, {"_id": 0})
        already_processed = tx and tx.get("status") == "paid"
        if status.payment_status == "paid" and not already_processed:
            # Extend pro_expires_at by 30 days
            current = await db.users.find_one({"user_id": user.user_id}, {"_id": 0})
            now = datetime.now(timezone.utc)
            existing_exp = None
            if current and current.get("pro_expires_at"):
                try:
                    existing_exp = datetime.fromisoformat(current["pro_expires_at"])
                    if existing_exp.tzinfo is None:
                        existing_exp = existing_exp.replace(tzinfo=timezone.utc)
                except Exception:
                    existing_exp = None
            base = existing_exp if existing_exp and existing_exp > now else now
            new_exp = base + timedelta(days=30)
            await db.users.update_one(
                {"user_id": user.user_id},
                {"$set": {"pro_expires_at": new_exp.isoformat()}},
            )
            await db.payment_transactions.update_one(
                {"session_id": session_id},
                {"$set": {"status": "paid", "paid_at": now.isoformat()}},
            )
        updated_user = await db.users.find_one({"user_id": user.user_id}, {"_id": 0})
        return {
            "payment_status": status.payment_status,
            "status": status.status,
            "is_pro": is_pro(User(**updated_user)),
            "pro_expires_at": updated_user.get("pro_expires_at"),
        }
    except Exception as e:
        logger.error(f"status error: {e}")
        raise HTTPException(500, f"Status check failed: {str(e)}")


# ---------- PRO TOOLS ----------
@api_router.post("/pro/diagnose")
async def pro_diagnose(req: ScanImageRequest, authorization: Optional[str] = Header(None)):
    user = await require_pro(authorization=authorization)
    try:
        chat = await _gemini_chat(
            session_id=f"diag-{uuid.uuid4()}",
            system="You are an expert appliance technician. Diagnose issues from photos. Return ONLY valid JSON.",
        )
        msg = UserMessage(
            text=(
                "Analyze this photo of a kitchen appliance. Identify any visible faults, wear, "
                "or issues (rust, frost build-up, dirty coils, broken parts, leaks, etc.). "
                'Return strict JSON: {"appliance":"...","status":"healthy|warning|critical",'
                '"issues":[{"name":"...","severity":"low|medium|high","explanation":"...","fix":"..."}],'
                '"maintenance_tips":["...","..."]}'
                ' If no appliance visible, return empty issues.'
            ),
            file_contents=[ImageContent(image_base64=req.image_base64)],
        )
        resp = await chat.send_message(msg)
        return json.loads(_strip_json(str(resp)))
    except Exception as e:
        logger.error(f"diagnose error: {e}")
        raise HTTPException(500, f"Diagnose failed: {str(e)}")


@api_router.post("/pro/rescue")
async def pro_rescue(req: ScanImageRequest, authorization: Optional[str] = Header(None)):
    user = await require_pro(authorization=authorization)
    try:
        chat = await _gemini_chat(
            session_id=f"rescue-{uuid.uuid4()}",
            system="You are a food rescue expert. You help revive wilted/about-to-go-bad food. Return ONLY valid JSON.",
        )
        msg = UserMessage(
            text=(
                "Analyze this photo of food (likely produce/leftovers that look wilted or aging). "
                "Suggest rescue/revival techniques AND recipes that work well with this condition. "
                'Return strict JSON: {"item":"...","condition":"fresh|wilting|near_spoilage|spoiled",'
                '"safe_to_use":true,"revival_tips":["...","..."],'
                '"rescue_recipes":[{"name":"...","why_it_works":"...","quick_steps":["...","..."]}]}'
            ),
            file_contents=[ImageContent(image_base64=req.image_base64)],
        )
        resp = await chat.send_message(msg)
        return json.loads(_strip_json(str(resp)))
    except Exception as e:
        logger.error(f"rescue error: {e}")
        raise HTTPException(500, f"Rescue failed: {str(e)}")


# Expiry tracker
@api_router.get("/pro/expiry", response_model=List[ExpiryItem])
async def list_expiry(authorization: Optional[str] = Header(None)):
    user = await require_pro(authorization=authorization)
    items = await db.expiry_items.find({"user_id": user.user_id}, {"_id": 0}).sort("expiry_date", 1).to_list(500)
    return [ExpiryItem(**i) for i in items]


@api_router.post("/pro/expiry", response_model=ExpiryItem)
async def add_expiry(item: ExpiryItemCreate, authorization: Optional[str] = Header(None)):
    user = await require_pro(authorization=authorization)
    obj = ExpiryItem(user_id=user.user_id, **item.dict())
    await db.expiry_items.insert_one(obj.dict())
    return obj


@api_router.delete("/pro/expiry/{item_id}")
async def delete_expiry(item_id: str, authorization: Optional[str] = Header(None)):
    user = await require_pro(authorization=authorization)
    res = await db.expiry_items.delete_one({"id": item_id, "user_id": user.user_id})
    return {"deleted": res.deleted_count}


# Voice chef (text chat for now)
@api_router.post("/pro/voice-chef")
async def voice_chef(req: VoiceChefMessage, authorization: Optional[str] = Header(None)):
    user = await require_pro(authorization=authorization)
    settings = await db.settings.find_one({"_id": "user"}, {"_id": 0}) or {}
    country = settings.get("country", "IN")
    _, cuisine = country_to_cuisine(country)
    try:
        chat = await _gemini_chat(
            session_id=f"vchef-{req.session_id}",
            system=(
                f"You are Chef Saanvi, a warm, friendly personal cooking assistant specializing in {cuisine} cuisine. "
                "Answer cooking questions concisely (2-4 sentences). Suggest ingredient swaps, "
                "technique tips, and timing. Be encouraging and conversational."
            ),
        )
        resp = await chat.send_message(UserMessage(text=req.message))
        text = str(resp)
        # Persist chat
        await db.voice_chef_messages.insert_many([
            {"session_id": req.session_id, "user_id": user.user_id, "role": "user",
             "text": req.message, "ts": datetime.now(timezone.utc).isoformat()},
            {"session_id": req.session_id, "user_id": user.user_id, "role": "assistant",
             "text": text, "ts": datetime.now(timezone.utc).isoformat()},
        ])
        return {"reply": text}
    except Exception as e:
        logger.error(f"voice_chef error: {e}")
        raise HTTPException(500, f"Voice chef failed: {str(e)}")


@api_router.get("/pro/voice-chef/{session_id}")
async def voice_chef_history(session_id: str, authorization: Optional[str] = Header(None)):
    user = await require_pro(authorization=authorization)
    msgs = await db.voice_chef_messages.find(
        {"session_id": session_id, "user_id": user.user_id}, {"_id": 0}
    ).sort("ts", 1).to_list(500)
    return {"messages": msgs}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    # Indexes
    await db.users.create_index("email", unique=True)
    await db.users.create_index("user_id", unique=True)
    await db.user_sessions.create_index("session_token", unique=True)
    await db.expiry_items.create_index([("user_id", 1), ("expiry_date", 1)])


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
