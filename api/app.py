from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Optional
from dotenv import load_dotenv
import uvicorn
import logging
import requests
import os
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

from perplexity_service import PerplexityService, get_perplexity_service, PerplexityAPIError, ParsingError

load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vercel Analytics 미들웨어
class VercelAnalyticsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if "text/html" in response.headers.get("content-type", ""):
            response.headers["x-vercel-analytics"] = "true"
        return response

# Initialize FastAPI app
app = FastAPI(title="Meditrek")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 환경에서는 특정 도메인만 허용하는 것이 좋습니다
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(VercelAnalyticsMiddleware)

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 서비스 의존성 설정
app.dependency_overrides[get_perplexity_service] = lambda: PerplexityService()

@app.get("/", response_class=HTMLResponse)
async def get_landing(request: Request):
    """Render the landing page"""
    return templates.TemplateResponse(
        "landing.html", 
        {"request": request}
    )


@app.get("/form", response_class=HTMLResponse)
async def get_form(request: Request):
    """Render the form page"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request):
    try:
        form_data = await request.form()
        symptoms = form_data.get("symptoms", "")
        gender = form_data.get("gender", "not specified")
        age = form_data.get("age", "not specified")
        allergic = form_data.get("allergic", "none")
        
        # Split and clean symptoms
        symptom_list = [s.strip() for s in symptoms.split(',') if s.strip()]
        
        if not symptom_list:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": "증상을 하나 이상 선택해주세요."
                }
            )
            
        perplexity_service = get_perplexity_service()
        medications, management_lists = perplexity_service.get_combined_recommendations(symptom_list, gender, age, allergic)
        
        logger.info(f"Received medications: {medications}")
        logger.info(f"Received management lists: {management_lists}")
        
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "medications": medications,
                "symptoms": symptoms,
                "gender": gender,
                "age": age,
                "allergic": allergic,
                "to_do_list": management_lists["to_do_list"],
                "do_not_list": management_lists["do_not_list"]
            }
        )
        
    except Exception as e:
        logger.exception("Error in recommend endpoint")
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_title": "오류가 발생했습니다",
                "error_message": "처리 중 문제가 발생했습니다. 다시 시도해주세요.",
                "error_detail": str(e),
                "debug_mode": os.getenv("DEBUG", "False").lower() == "true"
            },
            status_code=500
        )

@app.get("/api/pharmacies")
async def get_nearby_pharmacies(
    zipcode: str,
    perplexity_service: PerplexityService = Depends(get_perplexity_service)
    
):
    """Get nearby pharmacies based on zipcode"""
    try:
        api_key = os.getenv("GOOGLE_PLACES_API_KEY")
        
        if not api_key:
            logger.error("Google Places API key not found in environment variables")
            return JSONResponse(
                status_code=500,
                content={"error": "API key not configured. Please contact the administrator."}
            )
        
        logger.info(f"Searching pharmacies for zipcode: {zipcode}")
        
        # Google Geocoding API call to convert zipcode to coordinates
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zipcode}&key={api_key}"
        geocode_response = requests.get(geocode_url, timeout=10)
        
        if geocode_response.status_code != 200:
            logger.error(f"Geocode API HTTP error: {geocode_response.status_code}")
            return JSONResponse(
                status_code=geocode_response.status_code,
                content={"error": f"Error connecting to geocoding service: {geocode_response.status_code}"}
            )
        
        geocode_data = geocode_response.json()
        
        if geocode_data["status"] != "OK":
            logger.error(f"Geocode API error: {geocode_data.get('status')}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Geocoding service error: {geocode_data.get('status')}"}
            )
        
        if not geocode_data["results"]:
            logger.error("Geocode API returned no results")
            return JSONResponse(
                status_code=404,
                content={"error": "Location not found for this ZIP code"}
            )
        
        # Extract location data
        location = geocode_data["results"][0]["geometry"]["location"]
        lat, lng = location["lat"], location["lng"]
        logger.info(f"Location found - latitude: {lat}, longitude: {lng}")
        
        # Google Places API call to find nearby pharmacies
        places_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=5000&type=pharmacy&key={api_key}"
        places_response = requests.get(places_url, timeout=10)
        
        if places_response.status_code != 200:
            logger.error(f"Places API HTTP error: {places_response.status_code}")
            return JSONResponse(
                status_code=places_response.status_code,
                content={"error": f"Error connecting to places service: {places_response.status_code}"}
            )
        
        places_data = places_response.json()
        
        if places_data["status"] != "OK":
            logger.error(f"Places API error: {places_data.get('status')}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Places service error: {places_data.get('status')}"}
            )
        
        if not places_data["results"]:
            logger.warning("No pharmacies found near this location")
            return JSONResponse(
                status_code=404,
                content={"error": "No pharmacies found near this location"}
            )
        
        # Process pharmacy data
        pharmacies = []
        for place in places_data["results"][:3]:
            pharmacy = {
                "name": place["name"],
                "address": place.get("vicinity", "Address not available"),
                "distance": "Nearby"  # For exact distance calculation, we'd need to use Distance Matrix API
            }
            pharmacies.append(pharmacy)
        
        logger.info(f"Found {len(pharmacies)} pharmacies near {zipcode}")
        return JSONResponse(content={"pharmacies": pharmacies})
        
    except requests.exceptions.Timeout:
        logger.error("Request to Google API timed out")
        return JSONResponse(
            status_code=504,
            content={"error": "Request to Google API timed out"}
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Error connecting to Google API"}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred"}
        )
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# 앱 설정 부분 (app 변수 설정 후)
@app.exception_handler(PerplexityAPIError)
async def perplexity_api_exception_handler(request: Request, exc: PerplexityAPIError):
    """Handle Perplexity API errors"""
    logger.error(f"Perplexity API Error: {exc}")
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_title": "Service Temporarily Unavailable",
            "error_message": "Our recommendation service is temporarily unavailable. Please try again in a few moments.",
            "error_detail": str(exc),
            "debug_mode": os.getenv("DEBUG", "False").lower() == "true"
        },
        status_code=503
    )

@app.exception_handler(ParsingError)
async def parsing_exception_handler(request: Request, exc: ParsingError):
    """Handle parsing errors"""
    logger.error(f"Parsing Error: {exc}")
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_title": "Cannot Process Results",
            "error_message": "We had trouble processing the information. Please try again with different symptoms.",
            "error_detail": str(exc),
            "debug_mode": os.getenv("DEBUG", "False").lower() == "true"
        },
        status_code=422
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled Exception: {exc}")
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_title": "Something Went Wrong",
            "error_message": "An unexpected error occurred. Our team has been notified.",
            "error_detail": str(exc),
            "debug_mode": os.getenv("DEBUG", "False").lower() == "true"
        },
        status_code=500
    )

# Server startup code
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)