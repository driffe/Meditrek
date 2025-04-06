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

    def query_perplexity(self, query: str, max_retries: int = 1, timeout: int = 8) -> Optional[str]:
        """Send a query to the Perplexity API using requests."""
        start_time = time.time()
        logger.info(f"Starting API request for query: {query[:100]}...")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 쿼리 길이 제한
        if len(query) > 1000:
            query = query[:1000] + "..."
        
        payload = {
            "model": "sonar",  # 더 가벼운 모델 사용
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
        
        try:
            request_start = time.time()
            logger.info("Sending query to Perplexity API")
            
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
            logger.error(f"API request timed out: {e}")
            return None
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            return None
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            return None
    
    def get_combined_recommendations(self, symptoms: List[str], gender: str, age: str, allergic: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """Get both medication recommendations and management lists in a single API call."""
        symptoms_text = ", ".join(symptoms)
        query = (
            f"As a medical professional, provide recommendations for a {age} year old {gender} "
            f"with allergies to {allergic} who has the following symptoms: {symptoms_text}.\n\n"
            f"Format your response EXACTLY as follows:\n\n"
            f"MEDICATIONS:\n"
            f"1. Brand name: [medication name]\n"
            f"Form: [pill/tablet/liquid/gel/capsule/cream/ointment/lotion]\n"
            f"Side effects: [list main side effects]\n\n"
            f"2. Brand name: [medication name]\n"
            f"Form: [pill/tablet/liquid/gel/capsule/cream/ointment/lotion]\n"
            f"Side effects: [list main side effects]\n\n"
            f"3. Brand name: [medication name]\n"
            f"Form: [pill/tablet/liquid/gel/capsule/cream/ointment/lotion]\n"
            f"Side effects: [list main side effects]\n\n"
            f"MANAGEMENT:\n"
            f"DO:\n"
            f"1. [action]\n"
            f"2. [action]\n"
            f"3. [action]\n\n"
            f"DON'T:\n"
            f"1. [action]\n"
            f"2. [action]\n"
            f"3. [action]"
        )
        
        logger.info(f"=== Combined Recommendations API Call ===")
        logger.info(f"Starting combined query for symptoms: {symptoms_text}")
        
        # 캐시 키 생성 및 확인
        cache_key = self._get_cache_key(query)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            logger.info("Using cached combined response")
            return self._parse_combined_response(cached_response)
        
        start_time = time.time()
        response_text = self.query_perplexity(query)  # 기본 타임아웃 설정 사용
        total_time = time.time() - start_time
        
        logger.info(f"Combined API Total Time: {total_time:.2f}s")
        logger.info(f"=== Combined API Call End ===")
        
        if not response_text:
            logger.error("No response received from Perplexity API")
            return [], {"to_do_list": [], "do_not_list": []}
            
        # 응답 캐시에 저장
        self._cache_response(cache_key, response_text)
        logger.info(f"Received combined response: {response_text}")
        return self._parse_combined_response(response_text)
        
    def _parse_combined_response(self, response_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """Parse the combined response into medications and management lists."""
        try:
            logger.info("Starting to parse combined response...")
            
            # Split response into medications and management sections
            sections = response_text.split("MANAGEMENT:")
            if len(sections) != 2:
                logger.error("Could not split response into medications and management sections")
                return [], {"to_do_list": [], "do_not_list": []}
                
            medications_text, management_text = sections
            
            # Parse medications
            medications = self.parse_medication_recommendations(medications_text.replace("MEDICATIONS:", "").strip())
            
            # Parse management lists
            management_lists = self.parse_management_lists(management_text.strip())
            
            return medications, management_lists
            
        except Exception as e:
            logger.exception(f"Error parsing combined response: {e}")
            return [], {"to_do_list": [], "do_not_list": []}
    
    def parse_medication_recommendations(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract medication recommendations from Perplexity response."""
        medications = []
        try:
            # Remove ** characters from the response
            cleaned_text = response_text.replace('*', '')
            
            # Split recommendations by numbered items
            medication_sections = re.split(r'\n\s*\d+\.\s*', cleaned_text)
            medication_sections = [s.strip() for s in medication_sections if s.strip()]
            rank = 1

            for section in medication_sections[:3]:  # Process max 3 items
                if not section.strip() or "none recommended" in section.lower():
                    continue

                medication_info = {
                    "rank": rank,
                    "name": None,
                    "medication_type": None,
                    "side_effects": "Not available",
                }

                # Extract brand name with generic name in parentheses
                name_match = re.search(r'brand name:?\s*([^:\n]+)', section, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1).strip()
                    medication_info["name"] = name

                # Extract medication form
                form_match = re.search(r'form:?\s*([^:\n]+)', section, re.IGNORECASE)
                if form_match:
                    medication_form = form_match.group(1).strip()
                    medication_info["medication_type"] = medication_form

                # Extract side effects
                side_effects_match = re.search(r'side effects:?\s*([^\n]+(?:\n\s+[^\n]+)*)', section, re.IGNORECASE)
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
                logger.error(f"No medications found in response: {cleaned_text}")

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

    def parse_management_lists(self, response_text: str) -> Dict[str, List[str]]:
        """Parse the response text into to-do list and do-not list."""
        result = {
            "to_do_list": [],
            "do_not_list": []
        }
        
        try:
            # Remove ** characters from the response
            cleaned_text = response_text.replace('*', '')
            logger.info(f"Cleaned response text: {cleaned_text}")

            # First try to find DO section with more flexible pattern
            do_match = re.search(r'DO:[\s\n]*((?:\d+\.[^\n]+\n?)+)', cleaned_text, re.IGNORECASE)
            if do_match:
                do_section = do_match.group(1)
                logger.info(f"Found DO section: {do_section}")
                do_items = re.findall(r'\d+\.\s*([^\n]+)', do_section)
                logger.info(f"Extracted DO items: {do_items}")
                
                # Clean up items and limit to 3
                cleaned_items = []
                for item in do_items[:3]:
                    if item.strip():
                        # Remove reference numbers and special characters
                        item = re.sub(r'\[\d+\](?:\[\d+\])*', '', item)
                        item = re.sub(r'[^\w\s.,()-]', '', item)
                        cleaned_items.append(item.strip())
                result["to_do_list"] = cleaned_items
                logger.info(f"Final DO items: {result['to_do_list']}")
            else:
                logger.warning("Could not find DO section in response")

            # Then try to find DON'T section with more flexible pattern
            dont_match = re.search(r"DON'?T:[\s\n]*((?:\d+\.[^\n]+\n?)+)", cleaned_text, re.IGNORECASE)
            if dont_match:
                dont_section = dont_match.group(1)
                logger.info(f"Found DON'T section: {dont_section}")
                dont_items = re.findall(r'\d+\.\s*([^\n]+)', dont_section)
                logger.info(f"Extracted DON'T items: {dont_items}")
                
                # Clean up items and limit to 3
                cleaned_items = []
                for item in dont_items[:3]:
                    if item.strip():
                        # Remove reference numbers and special characters
                        item = re.sub(r'\[\d+\](?:\[\d+\])*', '', item)
                        item = re.sub(r'[^\w\s.,()-]', '', item)
                        cleaned_items.append(item.strip())
                result["do_not_list"] = cleaned_items
                logger.info(f"Final DON'T items: {result['do_not_list']}")
            else:
                logger.warning("Could not find DON'T section in response")

            return result
            
        except Exception as e:
            logger.exception(f"Error parsing management lists: {e}")
            logger.error(f"Failed response text: {response_text}")
            return result