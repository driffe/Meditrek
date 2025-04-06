import os
import requests
import json
import re
import time
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import logging

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# 의존성 주입을 위한 함수
def get_perplexity_service():
    """Dependency injection provider for PerplexityService"""
    return PerplexityService()

class PerplexityAPIError(Exception):
    """Perplexity API 호출 중 발생하는 오류"""
    pass

class ParsingError(Exception):
    """응답 파싱 중 발생하는 오류"""
    pass

class PerplexityService:
    """
    Perplexity API service for medication recommendations
    """
    
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.api_url = "https://api.perplexity.ai"
        self.last_response = None
        
        if not self.api_key:
            logger.warning("Perplexity API key not found in environment variables")
    
    def query_perplexity(self, query: str, max_retries: int = 3, timeout: int = 15) -> Optional[str]:
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
        
        retries = 0
        while retries < max_retries:
            try:
                logger.info(f"Sending query to Perplexity API (attempt {retries+1}/{max_retries})")
                response = requests.post(
                    f"{self.api_url}/chat/completions", 
                    headers=headers, 
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                
                # Extract the response text
                data = response.json()
                response_text = data["choices"][0]["message"]["content"]
                
                # Store the last response for debugging
                self.last_response = response_text
                logger.info(f"Received response from Perplexity API")
                
                return response_text
                
            except requests.exceptions.Timeout as e:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                logger.warning(f"Attempt {retries}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP Error: {e}")
                raise PerplexityAPIError(f"HTTP Error: {e}")
                
            except requests.exceptions.ConnectionError as e:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                logger.warning(f"Connection error (attempt {retries}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Unexpected Error: {e}")
                raise PerplexityAPIError(f"Unexpected Error: {e}")
        
        logger.error(f"Failed after {max_retries} retries")
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
                    r'^(?:\d+\.\s*)?([^:\n]+)(?::|$)',
                    r'(\w+(?:\s+\w+)*\s*\([^)]*)'  
                ]

                for pattern in name_patterns:
                    name_match = re.search(pattern, section, re.IGNORECASE | re.MULTILINE)
                    if name_match:
                        medication_name = name_match.group(1).strip()
                        # 괄호가 닫히지 않은 경우 처리
                        if '(' in medication_name and ')' not in medication_name:
                            # 괄호를 제거하거나 괄호 닫기 추가
                            if medication_name.count('(') == 1:
                                medication_name = medication_name.split('(')[0].strip()
                        medication_info["name"] = medication_name
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

            # 디버깅: 이름이 없는 약물이 있는지 확인
            for i, med in enumerate(medications):
                if not med.get('name'):
                    logger.warning(f"Medication at index {i} has no name: {med}")
                    # 기본 이름 설정
                    med['name'] = f"Medication {med.get('rank', i+1)}"

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

    def get_symptom_management_lists(self, symptoms: List[str]) -> Dict[str, List[str]]:
        """Get to-do list and do-not list based on symptoms."""
        symptoms_text = ", ".join(symptoms)
        
        query = (
            f"I have the following symptoms: {symptoms_text}. "
            f"Please provide two lists: "
            f"1. A list of things I SHOULD do to manage these symptoms (to-do list) "
            f"2. A list of things I should NOT do (do-not list) "
            f"Format the response as: "
            f"TO-DO LIST:\n"
            f"1. [item]\n"
            f"2. [item]\n"
            f"...\n"
            f"DO-NOT LIST:\n"
            f"1. [item]\n"
            f"2. [item]\n"
            f"..."
        )
        
        response_text = self.query_perplexity(query)
        if not response_text:
            logger.error("No response from Perplexity API for management lists")
            return {"to_do_list": [], "do_not_list": []}
        
        return self.parse_management_lists(response_text)
    
    def parse_management_lists(self, response_text: str) -> Dict[str, List[str]]:
        """Parse the response text into to-do list and do-not list."""
        result = {
            "to_do_list": [],
            "do_not_list": []
        }
        
        try:
            # Split the response into to-do and do-not sections
            sections = response_text.split("DO-NOT LIST:")
            if len(sections) != 2:
                return result
                
            to_do_section = sections[0].replace("TO-DO LIST:", "").strip()
            do_not_section = sections[1].strip()
            
            # Parse to-do list
            to_do_items = re.findall(r'\d+\.\s*([^\n]+)', to_do_section)
            result["to_do_list"] = [item.strip() for item in to_do_items if item.strip()]
            
            # Parse do-not list
            do_not_items = re.findall(r'\d+\.\s*([^\n]+)', do_not_section)
            result["do_not_list"] = [item.strip() for item in do_not_items if item.strip()]
            
            return result
            
        except Exception as e:
            logger.exception(f"Error parsing management lists: {e}, response_text: {response_text}")
            return result