"""
Localization Engine for cultural adaptation using AI.
"""
import os
import json 
import re


class LocalizationEngine:
    """
    Prompt-based cultural adaptation system for content localization.
    """
    
    def __init__(self, api_key=None, provider="openai"):
        """
        Initialize the localization engine.
        
        Args:
            api_key: API key for OpenAI
            provider: AI provider - 'openai' or 'huggingface'
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.provider = provider
    
    def localize(self, image_description, ocr_text, user_caption, language, region, tone):
        """
        Generate culturally adapted content.
        
        Args:
            image_description: Generated caption from image
            ocr_text: Extracted text from OCR
            user_caption: Optional user-provided caption
            language: Target language code (e.g., 'es', 'fr')
            region: Target region (e.g., 'mx', 'fr')
            tone: Desired tone (formal, casual, professional, friendly)
            
        Returns:
            dict: Localized content with cultural notes
        """
        try:
            # Build the prompt
            prompt = self._build_prompt(
                image_description=image_description,
                ocr_text=ocr_text,
                user_caption=user_caption,
                language=language,
                region=region,
                tone=tone
            )
            
            # Get response from AI
            response = self._generate_response(prompt)
            
            # Parse the response
            result = self._parse_response(response)
            
            return {
                'success': True,
                'localized_caption': result.get('localized_caption', ''),
                'localized_ui_text': result.get('localized_ui_text', ''),
                'cultural_notes': result.get('cultural_notes', []),
                'warnings': result.get('warnings', []),
                'language': language,
                'region': region,
                'tone': tone
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'localized_caption': None,
                'localized_ui_text': None,
                'cultural_notes': [],
                'warnings': []
            }
    
    def _build_prompt(self, image_description, ocr_text, user_caption, language, region, tone):
        """
        Build the localization prompt.
        """
        # Handle empty values
        image_desc = image_description or "No description available"
        ocr = ocr_text or "No text detected in image"
        caption = user_caption or "No user caption provided"
        
        # Map language codes to full names
        language_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'zh': 'Chinese', 'ja': 'Japanese',
            'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi', 'ru': 'Russian',
            'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish', 'vi': 'Vietnamese',
            'th': 'Thai', 'id': 'Indonesian', 'ms': 'Malay', 'sv': 'Swedish'
        }
        
        # Map regions to descriptions
        region_names = {
            'us': 'United States', 'uk': 'United Kingdom', 'ca': 'Canada',
            'au': 'Australia', 'in': 'India', 'br': 'Brazil', 'mx': 'Mexico',
            'fr': 'France', 'de': 'Germany', 'es': 'Spain', 'it': 'Italy',
            'jp': 'Japan', 'cn': 'China', 'kr': 'South Korea', 'sa': 'Saudi Arabia',
            'ae': 'United Arab Emirates', 'za': 'South Africa', 'ng': 'Nigeria',
            'ke': 'Kenya', 'eg': 'Egypt'
        }
        
        target_lang = language_names.get(language, language)
        target_region = region_names.get(region, region)
        
        prompt = f"""You are a cultural localization expert specializing in adapting content for different cultures and regions.

Image Description:
{image_desc}

Extracted Text from Image:
{ocr}

User Caption:
{caption}

Target Language: {target_lang}
Target Region: {target_region}
Tone: {tone}

Task:
1. Generate a culturally adapted caption that sounds natural in {target_lang} as spoken in {target_region}
2. Replace any cultural references that may not be appropriate for the target region
3. Maintain the emotional tone of the original content
4. Make it sound like it was naturally written by a local
5. Adapt any UI text or labels if present

Provide your response in the following JSON format:
{{
    "localized_caption": "Your adapted caption here",
    "localized_ui_text": "Any adapted UI text here (or empty string if none)",
    "cultural_notes": ["Note 1", "Note 2"],
    "warnings": ["Any cultural sensitivity warnings (or empty array)"]
}}

Remember: Translate meaning, not word-for-word. Focus on cultural adaptation and natural-sounding localization.

Response:"""
        
        return prompt
    
    def _generate_response(self, prompt):
        """
        Generate response from AI provider.
        """
        if self.provider == "openai":
            return self._generate_openai(prompt)
        else:
            return self._generate_huggingface(prompt)
    
    def _generate_openai(self, prompt):
        """
        Generate response using OpenAI API.
        """
        try:
            from openai import OpenAI
            
            if not self.api_key:
                raise ValueError("OpenAI API key not provided")
            
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a cultural localization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _generate_huggingface(self, prompt):
        """
        Generate response using HuggingFace (fallback - requires local model).
        """
        try:
            from transformers import pipeline
            
            # Use a text generation model
            generator = pipeline("text-generation", model="google/flan-t5-base")
            
            response = generator(
                prompt,
                max_length=500,
                num_return_sequences=1,
                temperature=0.7
            )
            
            return response[0]['generated_text']
            
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        except Exception as e:
            raise Exception(f"HuggingFace error: {str(e)}")
    
    def _parse_response(self, response_text):
        """
        Parse the AI response into structured data.
        """
        try:
            # Try to extract JSON from response
            # First, look for JSON block
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                # If no JSON found, create a simple structure
                result = {
                    'localized_caption': response_text.strip(),
                    'localized_ui_text': '',
                    'cultural_notes': [],
                    'warnings': []
                }
            
            return result
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
            return {
                'localized_caption': response_text.strip(),
                'localized_ui_text': '',
                'cultural_notes': ['Could not parse structured response'],
                'warnings': []
            }
    
    def get_supported_languages(self):
        """
        Get list of supported languages.
        """
        return [
            {'code': 'en', 'name': 'English'},
            {'code': 'es', 'name': 'Spanish'},
            {'code': 'fr', 'name': 'French'},
            {'code': 'de', 'name': 'German'},
            {'code': 'it', 'name': 'Italian'},
            {'code': 'pt', 'name': 'Portuguese'},
            {'code': 'zh', 'name': 'Chinese'},
            {'code': 'ja', 'name': 'Japanese'},
            {'code': 'ko', 'name': 'Korean'},
            {'code': 'ar', 'name': 'Arabic'},
            {'code': 'hi', 'name': 'Hindi'},
            {'code': 'ru', 'name': 'Russian'},
            {'code': 'nl', 'name': 'Dutch'},
            {'code': 'pl', 'name': 'Polish'},
            {'code': 'tr', 'name': 'Turkish'},
            {'code': 'vi', 'name': 'Vietnamese'},
            {'code': 'th', 'name': 'Thai'},
            {'code': 'id', 'name': 'Indonesian'},
            {'code': 'ms', 'name': 'Malay'},
            {'code': 'sv', 'name': 'Swedish'}
        ]
    
    def get_supported_regions(self):
        """
        Get list of supported regions.
        """
        return [
            {'code': 'us', 'name': 'United States'},
            {'code': 'uk', 'name': 'United Kingdom'},
            {'code': 'ca', 'name': 'Canada'},
            {'code': 'au', 'name': 'Australia'},
            {'code': 'in', 'name': 'India'},
            {'code': 'br', 'name': 'Brazil'},
            {'code': 'mx', 'name': 'Mexico'},
            {'code': 'fr', 'name': 'France'},
            {'code': 'de', 'name': 'Germany'},
            {'code': 'es', 'name': 'Spain'},
            {'code': 'it', 'name': 'Italy'},
            {'code': 'jp', 'name': 'Japan'},
            {'code': 'cn', 'name': 'China'},
            {'code': 'kr', 'name': 'South Korea'},
            {'code': 'sa', 'name': 'Saudi Arabia'},
            {'code': 'ae', 'name': 'United Arab Emirates'},
            {'code': 'za', 'name': 'South Africa'},
            {'code': 'ng', 'name': 'Nigeria'},
            {'code': 'ke', 'name': 'Kenya'},
            {'code': 'eg', 'name': 'Egypt'}
        ]
    
    def get_supported_tones(self):
        """
        Get list of supported tones.
        """
        return ['formal', 'casual', 'professional', 'friendly']


def get_localization_engine(api_key=None, provider="openai"):
    """
    Factory function to get localization engine.
    """
    return LocalizationEngine(api_key=api_key, provider=provider)
