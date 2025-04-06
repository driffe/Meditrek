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
        self._cache = {}
        self._cache_timeout = 3600  # 1시간 캐시
        
        if not self.api_key:
            logger.warning("Perplexity API key not found in environment variables")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key"""
        return f"query_{hash(query)}"

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response"""
        if cache_key in self._cache:
            timestamp, response = self._cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                logger.info("Using cached response")
                return response
            else:
                del self._cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache response"""
        self._cache[cache_key] = (time.time(), response)

    def query_perplexity(self, query: str, max_retries: int = 2, timeout: int = 5) -> Optional[str]:
        """Send a query to the Perplexity API using requests."""
        start_time = time.time()
        logger.info(f"Starting API request for query: {query[:100]}...")

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
                request_start = time.time()
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
                request_time = time.time() - request_start
                total_time = time.time() - start_time
                
                logger.info(f"API Response Time: {request_time:.2f}s")
                logger.info(f"Total Processing Time: {total_time:.2f}s")
                logger.info(f"Response length: {len(response_text)} characters")
                
                return response_text
                
            except requests.exceptions.Timeout as e:
                retries += 1
                wait_time = 1  # 고정된 대기 시간으로 변경
                if retries < max_retries:
                    logger.warning(f"Attempt {retries}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Final attempt failed: {e}")
                    return None
                
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP Error: {e}")
                return None
                
            except requests.exceptions.ConnectionError as e:
                retries += 1
                wait_time = 1  # 고정된 대기 시간으로 변경
                if retries < max_retries:
                    logger.warning(f"Connection error (attempt {retries}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Final attempt failed: {e}")
                    return None
                
            except Exception as e:
                logger.error(f"Unexpected Error: {e}")
                return None
        
        logger.error(f"Failed after {max_retries} retries")
        return None
    
    def get_medication_recommendations(self, symptoms: List[str], gender: str, age: str, allergic: str) -> Optional[List[Dict[str, Any]]]:
        """Get medication recommendations based on symptoms, gender, age and allergies."""
        symptoms_text = ", ".join(symptoms)
        
        query = (
            f"As a medical professional, recommend 3 over-the-counter medications for a {age} year old {gender} "
            f"with allergies to {allergic} who has the following symptoms: {symptoms_text}. "
            f"Format your response exactly as follows for each medication:\n\n"
            f"1. Brand name: [medication name]\n"
            f"Form: [pill/tablet/liquid/gel/capsule/cream/ointment/lotion]\n"
            f"Side effects: [list main side effects]\n\n"
            f"2. Brand name: [medication name]\n"
            f"Form: [pill/tablet/liquid/gel/capsule/cream/ointment/lotion]\n"
            f"Side effects: [list main side effects]\n\n"
            f"3. Brand name: [medication name]\n"
            f"Form: [pill/tablet/liquid/gel/capsule/cream/ointment/lotion]\n"
            f"Side effects: [list main side effects]\n\n"
            f"Important: Provide ONLY the medication information in the exact format above. "
            f"Do not include any additional text, introductions, or explanations."
        )
        
        response_text = self.query_perplexity(query)
        if not response_text:
            return None
        
        return self.parse_medication_recommendations(response_text)
    
    def parse_medication_recommendations(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract medication recommendations from Perplexity response."""
        medications = []
        try:
            # Split recommendations by numbered items
            medication_sections = re.split(r'\n\s*\d+\.\s*', response_text)
            medication_sections = [s.strip() for s in medication_sections if s.strip()]
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

                # Extract brand name with generic name in parentheses
                name_match = re.search(r'brand name:\s*([^:\n]+)', section, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1).strip()
                    medication_info["name"] = name

                # Extract medication form
                form_match = re.search(r'form:\s*([^:\n]+)', section, re.IGNORECASE)
                if form_match:
                    medication_form = form_match.group(1).strip()
                    medication_info["medication_type"] = medication_form

                # Extract side effects
                side_effects_match = re.search(r'side effects:\s*([^\n]+(?:\n\s+[^\n]+)*)', section, re.IGNORECASE)
                if side_effects_match:
                    side_effects = side_effects_match.group(1).strip()
                    medication_info["side_effects"] = side_effects

                # Add medication if name is present
                if medication_info["name"]:
                    pharmacy_links = self.create_pharmacy_links(medication_info["name"])
                    medication_info.update(pharmacy_links)
                    medications.append(medication_info)
                    rank += 1
                else:
                    logger.warning(f"Skipping medication due to missing name: {section}")

            if not medications:
                logger.error(f"No medications found in response: {response_text}")

            return medications
        except Exception as e:
            logger.exception(f"Error parsing medication recommendations: {e}, response_text: {response_text}")
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
            #"walgreens_link": f"https://www.walgreens.com/search/results.jsp?Ntt={encoded_search}",
        }

    def get_symptom_management_lists(self, symptoms: List[str]) -> Dict[str, List[str]]:
        """Get to-do list and do-not list based on symptoms."""
        symptoms_text = ", ".join(symptoms)
        query = (
            f"As a medical professional, provide two lists for managing these symptoms: {symptoms_text}. "
            f"Format exactly as:\n\n"
            f"DO:\n"
            f"1. [action]\n"
            f"2. [action]\n"
            f"3. [action]\n"
            f"4. [action]\n"
            f"5. [action]\n\n"
            f"DON'T:\n"
            f"1. [action]\n"
            f"2. [action]\n"
            f"3. [action]\n"
            f"4. [action]\n"
            f"5. [action]"
        )
        
        logger.info(f"Sending management lists query for symptoms: {symptoms_text}")
        response_text = self.query_perplexity(query)
        
        if not response_text:
            logger.error("No response received from Perplexity API for management lists")
            return {"to_do_list": [], "do_not_list": []}
            
        logger.info(f"Received management lists response: {response_text}")
        return self.parse_management_lists(response_text)
    
    def parse_management_lists(self, response_text: str) -> Dict[str, List[str]]:
        """Parse the response text into to-do list and do-not list."""
        result = {
            "to_do_list": [],
            "do_not_list": []
        }
        
        try:
            logger.info(f"Parsing management lists from response: {response_text}")

            # First try to find DO section
            do_match = re.search(r'DO:\s*((?:\d+\.[^\n]+\n?)+)', response_text, re.IGNORECASE)
            if do_match:
                do_section = do_match.group(1)
                do_items = re.findall(r'\d+\.\s*([^\n]+)', do_section)
                # Clean up items: remove reference numbers and special characters
                cleaned_items = []
                for item in do_items:
                    if item.strip():
                        # Remove reference numbers like [1][7]
                        item = re.sub(r'\[\d+\](?:\[\d+\])*', '', item)
                        # Remove special characters but keep periods and commas
                        item = re.sub(r'[^\w\s.,()-]', '', item)
                        cleaned_items.append(item.strip())
                result["to_do_list"] = cleaned_items
                logger.info(f"Found {len(result['to_do_list'])} DO items: {result['to_do_list']}")
            else:
                logger.warning("Could not find DO section")

            # Then try to find DON'T section
            dont_match = re.search(r"DON'?T:\s*((?:\d+\.[^\n]+\n?)+)", response_text, re.IGNORECASE)
            if dont_match:
                dont_section = dont_match.group(1)
                dont_items = re.findall(r'\d+\.\s*([^\n]+)', dont_section)
                # Clean up items: remove reference numbers and special characters
                cleaned_items = []
                for item in dont_items:
                    if item.strip():
                        # Remove reference numbers like [1][7]
                        item = re.sub(r'\[\d+\](?:\[\d+\])*', '', item)
                        # Remove special characters but keep periods and commas
                        item = re.sub(r'[^\w\s.,()-]', '', item)
                        cleaned_items.append(item.strip())
                result["do_not_list"] = cleaned_items
                logger.info(f"Found {len(result['do_not_list'])} DON'T items: {result['do_not_list']}")
            else:
                logger.warning("Could not find DON'T section")

            if not result["to_do_list"] and not result["do_not_list"]:
                logger.error(f"No items found in either list. Full response: {response_text}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error parsing management lists: {e}")
            logger.error(f"Failed response text: {response_text}")
            return result