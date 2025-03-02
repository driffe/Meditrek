import os
import requests
import json
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import logging

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

class PerplexityService:
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.api_url = "https://api.perplexity.ai"
        
        if not self.api_key:
            logger.warning("Perplexity API key not found in environment variables")
    
    def query_perplexity(self, query: str) -> Optional[str]:
        """Send a query to the Perplexity API using requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions", 
                headers=headers, 
                json=payload
            )
            response.raise_for_status()
            
            # Extract the response text
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error querying Perplexity API: {e}")

            if 'response' in locals():
                logger.error(f"Response status: {response.status_code}")
                logger.error(f"Response text: {response.text}")
            return None
    
    def get_medication_recommendations(self, symptoms: List[str], gender: str, age: str, allergic: str) -> Optional[List[Dict[str, Any]]]:
        """Get medication recommendations based on symptoms, gender, age and allergies."""
        symptoms_text = ", ".join(symptoms)
        
        query = (
            f"I'm {gender} and {age} years old. "
            f"I'm allergic to {allergic}. "
            f"I have the following symptoms: {symptoms_text}. "
            f"Please recommend exactly 3 over-the-counter medications "
            f"that would help, ranked by effectiveness (1st, 2nd, and 3rd choice). "
            f"For each medication, provide: 1) Brand name, 2) Type of Medication ex) pill/tablet, Power, liquid/gel, Capsules, Creams/ointments/lotions 3) Side effects "
            f"Format as a list with these details for each medication. "
            f"Give list of medication without introduction ex) Here is list of medication or Based on the information provided"
        )
        
        response_text = self.query_perplexity(query)
        if not response_text:
            return None
        
        return self.parse_medication_recommendations(response_text)
    
    def parse_medication_recommendations(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract medication recommendations from Perplexity response."""
        medications = []
        try:
            # Split recommendations by rank or numbered items
            medication_sections = re.split(r'(?:\n\s*\n|\n\s*(?:\d+(?:st|nd|rd|th)\s*choice|choice\s*\d+:|^\d+\.))', response_text)
            medication_sections = [s for s in medication_sections if s.strip()]
            rank = 1

            for section in medication_sections[:3]:  # Process max 3 items
                if not section.strip():
                    continue

                medication_info = {
                    "rank": rank,
                    "name": None,
                    "medication_type": None,
                    "side_effects": "Not available",
                }

                # Extract medication name
                name_patterns = [
                    r'(?:brand name|medication|name):\s*([^\n]+)',
                    r'^(?:\d+\.\s*)?([^:\n]+)(?::|$)'
                ]

                for pattern in name_patterns:
                    name_match = re.search(pattern, section, re.IGNORECASE | re.MULTILINE)
                    if name_match:
                        medication_info["name"] = name_match.group(1).strip()
                        break

                # Extract medication type
                type_patterns = [
                    r'(?:type of medication|form):\s*([^\n]+)',
                    r'(?:pill|tablet|liquid|gel|capsule|cream|ointment|lotion|powder)s?'
                ]

                for pattern in type_patterns:
                    type_match = re.search(pattern, section, re.IGNORECASE)
                    if type_match:
                        if len(type_match.groups()) > 0:
                            medication_info["medication_type"] = type_match.group(1).strip()
                        else:
                            medication_info["medication_type"] = type_match.group(0).strip()
                        break

                # Extract side effects - More resilient approach
                side_effects_patterns = [
                    r'side effects:\s*([^\n]+(?:\n\s+[^\n]+)*)',  # Original pattern
                    r'side effects[^:]*?(?:include|are|:)\s*([^\n]+(?:\n\s+[^\n]+)*)',  # More general
                    r'(?:adverse effects|warnings):\s*([^\n]+(?:\n\s+[^\n]+)*)',  # Alternate terms
                    r'(?:may cause|can cause):\s*([^\n]+(?:\n\s+[^\n]+)*)'  # potential start of the phrase
                ]

                for pattern in side_effects_patterns:
                    side_effects_match = re.search(pattern, section, re.IGNORECASE)
                    if side_effects_match:
                        try:
                            # Remove Photo reference from side effects if present
                            side_effects_text = side_effects_match.group(1).strip()
                            medication_info["side_effects"] = side_effects_text
                            break  # Exit loop if a match is found
                        except IndexError as e:
                            logger.error(f"IndexError accessing regex group: {e}, pattern: {pattern}, section: {section}")

                if medication_info["name"]:
                    pharmacy_links = self.create_pharmacy_links(medication_info["name"])
                    medication_info.update(pharmacy_links)
                    medications.append(medication_info)
                    rank += 1

            return medications
        except Exception as e:
            logger.exception(f"Error parsing medication recommendations: {e}, response_text: {response_text}")  # Log full exception + response
            return []
        
    def create_pharmacy_links(self, medication_name):
        """Create pharmacy links for a medication"""
        # Clean up the medication name for search
        search_term = re.sub(r'\([^)]*\)', '', medication_name).strip()
        # Remove brand designations like "Extra Strength"
        search_term = re.sub(r'(?:extra strength|maximum strength|children\'s|infant\'s)', '', search_term, flags=re.IGNORECASE).strip()
        encoded_search = requests.utils.quote(search_term)
        
        return {
            "cvs_link": f"https://www.cvs.com/search?searchTerm={encoded_search}",
            "walgreens_link": f"https://www.walgreens.com/search/results.jsp?Ntt={encoded_search}",
        }