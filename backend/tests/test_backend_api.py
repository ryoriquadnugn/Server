"""Backend API tests for Fridge Chef (Pro tier + paywall + auth)."""
import pytest
import requests
import uuid

# --- Health ---
class TestHealth:
    def test_root(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/")
        assert r.status_code == 200, r.text
        j = r.json()
        assert j.get("status") == "ok"
        assert "message" in j


# --- Settings ---
class TestSettings:
    def test_get_settings_default(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/settings")
        assert r.status_code == 200, r.text
        j = r.json()
        assert "country" in j and "country_name" in j

    def test_update_settings_in(self, api_client, base_url):
        r = api_client.put(f"{base_url}/api/settings", json={"country": "IN", "country_name": "India"})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["country"] == "IN"
        assert "India" in j["country_name"]

        # Verify persistence
        r2 = api_client.get(f"{base_url}/api/settings")
        assert r2.status_code == 200
        assert r2.json()["country"] == "IN"

    def test_update_settings_us(self, api_client, base_url):
        r = api_client.put(f"{base_url}/api/settings", json={"country": "US", "country_name": "United States"})
        assert r.status_code == 200
        assert r.json()["country"] == "US"


# --- Subscription pricing ---
class TestSubscriptionPricing:
    def test_pricing_in(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/subscription/pricing", params={"country": "IN"})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["currency"] == "inr"
        assert j["amount"] == 100.0
        assert j["display"] == "₹100"
        assert j["country"] == "IN"
        assert j["interval"] == "month"

    def test_pricing_us(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/subscription/pricing", params={"country": "US"})
        assert r.status_code == 200
        j = r.json()
        assert j["currency"] == "usd"
        assert j["amount"] == 1.20
        assert j["display"] == "$1.20"

    def test_pricing_gb(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/subscription/pricing", params={"country": "GB"})
        assert r.status_code == 200
        j = r.json()
        assert j["currency"] == "gbp"
        assert j["amount"] == 0.95

    def test_pricing_unknown_defaults_to_in(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/subscription/pricing", params={"country": "ZZ"})
        assert r.status_code == 200
        assert r.json()["currency"] == "inr"


# --- Barcode (Open Food Facts) ---
class TestBarcode:
    def test_nutella_barcode(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/scan/barcode/3017620422003", timeout=20)
        assert r.status_code == 200, r.text
        j = r.json()
        assert j.get("found") is True
        assert j.get("barcode") == "3017620422003"
        name = (j.get("name") or "").lower()
        assert "nutella" in name, f"Expected Nutella in name, got: {j.get('name')}"

    def test_invalid_barcode(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/scan/barcode/0000000000000", timeout=20)
        # Either found false or 404; ensure no 500
        assert r.status_code in (200, 404), r.text


# --- Pantry CRUD ---
class TestPantry:
    def test_pantry_crud(self, api_client, base_url):
        # Clear
        api_client.delete(f"{base_url}/api/pantry")
        # Add
        payload = {"name": "TEST_tomato", "quantity": "2", "source": "manual"}
        r = api_client.post(f"{base_url}/api/pantry", json=payload)
        assert r.status_code == 200, r.text
        item = r.json()
        assert item["name"] == "TEST_tomato"
        assert "id" in item
        assert "_id" not in item
        item_id = item["id"]

        # List
        r = api_client.get(f"{base_url}/api/pantry")
        assert r.status_code == 200
        items = r.json()
        assert any(i["id"] == item_id for i in items)
        for i in items:
            assert "_id" not in i

        # Delete
        r = api_client.delete(f"{base_url}/api/pantry/{item_id}")
        assert r.status_code == 200
        assert r.json()["deleted"] == 1


# --- Recipe generation (AI - country localized) ---
class TestRecipes:
    def test_recipes_country_in(self, api_client, base_url):
        r = api_client.post(
            f"{base_url}/api/recipes/generate",
            json={"ingredients": ["onion", "tomato", "rice", "garlic"], "country": "IN", "count": 3},
            timeout=90,
        )
        assert r.status_code == 200, r.text
        recipes = r.json()
        assert isinstance(recipes, list) and len(recipes) > 0, "No recipes returned"
        for rec in recipes:
            assert "_id" not in rec
            cuisine = (rec.get("cuisine") or "").lower()
            country = (rec.get("country") or "").lower()
            assert "indian" in cuisine or "india" in country or rec.get("country", "").upper() == "IN", (
                f"Expected Indian cuisine, got cuisine={rec.get('cuisine')}, country={rec.get('country')}"
            )
            # No American cuisine leakage
            assert "american" not in cuisine, f"American cuisine leaked: {rec.get('title')}"

    def test_recipes_country_it(self, api_client, base_url):
        r = api_client.post(
            f"{base_url}/api/recipes/generate",
            json={"ingredients": ["pasta", "tomato", "basil", "garlic"], "country": "IT", "count": 2},
            timeout=90,
        )
        assert r.status_code == 200, r.text
        recipes = r.json()
        assert isinstance(recipes, list) and len(recipes) > 0
        for rec in recipes:
            cuisine = (rec.get("cuisine") or "").lower()
            country = (rec.get("country") or "").lower()
            assert "italian" in cuisine or "italy" in country or rec.get("country", "").upper() == "IT", (
                f"Expected Italian cuisine, got: {rec.get('cuisine')} / {rec.get('country')}"
            )

    def test_recipes_no_ingredients(self, api_client, base_url):
        r = api_client.post(
            f"{base_url}/api/recipes/generate",
            json={"ingredients": [], "country": "IN", "count": 1},
        )
        assert r.status_code == 400


# --- Auth (gated endpoints) ---
class TestAuthGating:
    def test_auth_me_no_token(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/auth/me")
        assert r.status_code == 401, r.text

    def test_auth_me_invalid_token(self, api_client, base_url):
        r = api_client.get(
            f"{base_url}/api/auth/me",
            headers={"Authorization": "Bearer invalid_token_xyz"},
        )
        assert r.status_code == 401

    def test_google_session_invalid(self, api_client, base_url):
        r = api_client.post(
            f"{base_url}/api/auth/google/session",
            json={"session_id": f"invalid_{uuid.uuid4().hex}"},
        )
        assert r.status_code == 401, r.text

    def test_checkout_requires_auth(self, api_client, base_url):
        r = api_client.post(
            f"{base_url}/api/subscription/checkout",
            json={"country": "IN", "origin_url": "https://example.com"},
        )
        assert r.status_code == 401, r.text

    def test_pro_diagnose_requires_auth(self, api_client, base_url):
        r = api_client.post(f"{base_url}/api/pro/diagnose", json={"image_base64": "x"})
        assert r.status_code == 401, r.text

    def test_pro_rescue_requires_auth(self, api_client, base_url):
        r = api_client.post(f"{base_url}/api/pro/rescue", json={"image_base64": "x"})
        assert r.status_code == 401, r.text

    def test_pro_expiry_get_requires_auth(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/pro/expiry")
        assert r.status_code == 401, r.text

    def test_pro_expiry_post_requires_auth(self, api_client, base_url):
        r = api_client.post(
            f"{base_url}/api/pro/expiry",
            json={"name": "milk", "expiry_date": "2026-12-31"},
        )
        assert r.status_code == 401

    def test_pro_voice_chef_requires_auth(self, api_client, base_url):
        r = api_client.post(
            f"{base_url}/api/pro/voice-chef",
            json={"session_id": "s1", "message": "hi"},
        )
        assert r.status_code == 401, r.text

    def test_pro_voice_chef_history_requires_auth(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/pro/voice-chef/s1")
        assert r.status_code == 401


# --- Response cleanliness: no Mongo _id leak ---
class TestResponseCleanliness:
    def test_settings_no_id(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/settings")
        assert r.status_code == 200
        assert "_id" not in r.json()

    def test_pricing_no_id(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/subscription/pricing", params={"country": "IN"})
        assert "_id" not in r.json()

    def test_pantry_no_id(self, api_client, base_url):
        r = api_client.get(f"{base_url}/api/pantry")
        for i in r.json():
            assert "_id" not in i
