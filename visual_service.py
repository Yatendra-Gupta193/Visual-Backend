"""
Visual Localization Service - Main business logic orchestrating all services.
"""
import argparse
import os
import tempfile
from PIL import Image

from captioning import get_captioning_service
from ocr_service import get_ocr_service
from localization_engine import get_localization_engine


class VisualLocalizationService:
    """
    Main service for Visual Localization that orchestrates:
    1. Image Captioning (OpenAI Vision)
    2. OCR (EasyOCR)
    3. Cultural Localization (AI)
    """
    
    def __init__(self, config=None):
        """
        Initialize the Visual Localization Service.
        
        Args:
            config: Optional configuration dictionary        
        """
        self.config = config or {}
        
        self.captioning_service = None
        self.ocr_service = None
        self.localization_engine = None
        
        self.captioning_provider = self.config.get('captioning_provider', 'openai')
        self.ocr_provider = self.config.get('ocr_provider', 'easyocr')
        self.localization_provider = self.config.get('localization_provider', 'openai')
        self.api_key = self.config.get('api_key', os.getenv('OPENAI_API_KEY'))
        self.use_gpu = self.config.get('use_gpu', False)
        self.ocr_languages = self.config.get('ocr_languages', ['en'])
    
    def _get_captioning_service(self):
        if self.captioning_service is None:
            self.captioning_service = get_captioning_service(
                provider=self.captioning_provider,
                api_key=self.api_key
            )
        return self.captioning_service
    
    def _get_ocr_service(self):
        if self.ocr_service is None:
            self.ocr_service = get_ocr_service(
                provider=self.ocr_provider,
                languages=self.ocr_languages,
                use_gpu=self.use_gpu
            )
        return self.ocr_service
    
    def _get_localization_engine(self):
        if self.localization_engine is None:
            self.localization_engine = get_localization_engine(
                api_key=self.api_key,
                provider=self.localization_provider
            )
        return self.localization_engine
    
    def process_image(self, image_file, target_language='en', target_region='us', 
                      tone='friendly', user_caption=None, ocr_language='en'):
        """
        Process an image and generate culturally adapted content.
        """
        result = {
            'success': False,
            'image_description': None,
            'ocr_text': None,
            'localized_caption': None,
            'localized_ui_text': None,
            'cultural_notes': [],
            'warnings': [],
            'language': target_language,
            'region': target_region,
            'tone': tone
        }
        
        temp_path = None
        try:
            if hasattr(image_file, 'save'):
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, image_file.filename)
                image_file.save(temp_path)
            elif isinstance(image_file, str):
                temp_path = image_file
            else:
                result['error'] = 'Invalid image file format'
                return result
            
            if ocr_language != self.ocr_languages[0]:
                self.ocr_languages = [ocr_language]
                self.ocr_service = None
            
            caption_result = self._generate_caption(temp_path)
            if caption_result['success']:
                result['image_description'] = caption_result['caption']
            else:
                result['error'] = f"Captioning failed: {caption_result.get('error', 'Unknown error')}"
                return result
            
            ocr_result = self._extract_text(temp_path)
            if ocr_result['success']:
                result['ocr_text'] = ocr_result['text']
            else:
                result['ocr_text'] = ''
            
            localization_result = self._localize_content(
                image_description=result['image_description'],
                ocr_text=result['ocr_text'],
                user_caption=user_caption,
                language=target_language,
                region=target_region,
                tone=tone
            )
            
            if localization_result['success']:
                result['localized_caption'] = localization_result['localized_caption']
                result['localized_ui_text'] = localization_result.get('localized_ui_text', '')
                result['cultural_notes'] = localization_result.get('cultural_notes', [])
                result['warnings'] = localization_result.get('warnings', [])
                result['success'] = True
            else:
                result['error'] = f"Localization failed: {localization_result.get('error', 'Unknown error')}"
                return result
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
        
        finally:
            if temp_path and temp_path != image_file and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def _generate_caption(self, image_path):
        service = self._get_captioning_service()
        return service.generate_caption(image_path)
    
    def _extract_text(self, image_path):
        service = self._get_ocr_service()
        return service.extract_text(image_path, detail_level='normal')
    
    def _localize_content(self, image_description, ocr_text, user_caption, 
                         language, region, tone):
        engine = self._get_localization_engine()
        return engine.localize(
            image_description=image_description,
            ocr_text=ocr_text,
            user_caption=user_caption,
            language=language,
            region=region,
            tone=tone
        )
    
    def cleanup(self):
        if self.captioning_service:
            self.captioning_service.cleanup()
        if self.ocr_service:
            self.ocr_service.cleanup()
    
    def get_supported_config(self):
        engine = self._get_localization_engine()
        return {
            'languages': engine.get_supported_languages(),
            'regions': engine.get_supported_regions(),
            'tones': engine.get_supported_tones()
        }


def get_visual_localization_service(config=None):
    return VisualLocalizationService(config=config)



    


