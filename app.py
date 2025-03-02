from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Optional
from dotenv import load_dotenv
import uvicorn
import logging
import requests
import os

from perplexity_service import PerplexityService

load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Medication Recommender")

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Perplexity service
perplexity_service = PerplexityService()

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
async def recommend_medication(
    request: Request,
    symptoms: str = Form(...),
    gender: str = Form(None),  # Make optional
    age: str = Form(None),     # Make optional
    allergic: str = Form(None)  # Make optional
):
    gender = gender or "not specified"
    age = age or "not specified"
    allergic = allergic or "none"
    """Process medication recommendation request"""
    try:
        # Split and clean symptoms
        symptom_list = [s.strip() for s in symptoms.split(',') if s.strip()]
        
        if not symptom_list:
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request,
                    "error": "Please enter at least one symptom."
                }
            )
        
        # Get medication recommendations via Perplexity API
        medications = perplexity_service.get_medication_recommendations(
            symptom_list, 
            gender, 
            age, 
            allergic
        )
        
        if not medications:
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request,
                    "error": "Failed to get medication recommendations. Please try again."
                }
            )
        
        # Render results page
        return templates.TemplateResponse(
            "results.html", 
            {
                "request": request,
                "medications": medications,
                "symptoms": symptoms,
                "gender": gender,
                "age": age,
                "allergic": allergic
            }
        )
        
    except Exception as e:
        logger.error(f"Error in medication recommendation: {e}")
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "error": "An error occurred while processing your request. Please try again."
            }
        )

@app.get("/api/pharmacies")
async def get_nearby_pharmacies(zipcode: str):
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

# Server startup code
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)