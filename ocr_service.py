"""
OCR Service using EasyOCR for text extraction from images.
"""
import easyocr
import numpy as np
from PIL import Image
import os


class OCRService:
    """
    Service for extracting text from images using EasyOCR.
    """
    
    def __init__(self, languages=['en'], use_gpu=False):
        """
        Initialize the OCR service.
        
        Args:
            languages: List of language codes (e.g., ['en'], ['en', 'es'], ['en', 'ja'])
            use_gpu: Whether to use GPU for OCR (if available)
        """
        self.languages = languages
        self.use_gpu = use_gpu
        self.reader = None
    
    def _load_reader(self):
        """Lazy load the EasyOCR reader only when needed."""
        if self.reader is None:
            print(f"Loading EasyOCR reader for languages: {self.languages}")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False
            )
            print("EasyOCR reader loaded successfully")
    
    def extract_text(self, image_path, detail_level='normal'):
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            detail_level: 'simple' (just text), 'normal' (with confidence), 
                         'detailed' (with bounding boxes)
            
        Returns:
            dict: OCR result with extracted text
        """
        try:
            # Load reader if not already loaded
            self._load_reader()
            
            # Read image
            image = Image.open(image_path)
            
            # Perform OCR
            results = self.reader.readtext(np.array(image))
            
            return self._process_results(results, detail_level)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': None,
                'all_text': []
            }
    
    def extract_text_from_image(self, image, detail_level='normal'):
        """
        Extract text from a PIL Image object.
        
        Args:
            image: PIL Image object
            detail_level: 'simple', 'normal', or 'detailed'
            
        Returns:
            dict: OCR result with extracted text
        """
        try:
            # Load reader if not already loaded
            self._load_reader()
            
            # Perform OCR
            results = self.reader.readtext(np.array(image))
            
            return self._process_results(results, detail_level)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': None,
                'all_text': []
            }
    
    def _process_results(self, results, detail_level):
        """
        Process OCR results based on detail level.
        
        Args:
            results: Raw EasyOCR results
            detail_level: Level of detail to return
            
        Returns:
            dict: Processed OCR result
        """
        all_text = []
        detailed_results = []
        
        for (bbox, text, confidence) in results:
            text = text.strip()
            if text:
                all_text.append(text)
                
                if detail_level in ['normal', 'detailed']:
                    detailed_results.append({
                        'text': text,
                        'confidence': round(confidence, 4),
                        'bbox': bbox if detail_level == 'detailed' else None
                    })
        
        combined_text = ' '.join(all_text)
        
        return {
            'success': True,
            'text': combined_text,
            'all_text': all_text,
            'detailed_results': detailed_results if detail_level != 'simple' else [],
            'language': self.languages,
            'text_count': len(all_text)
        }
    
    def extract_text_with_layout(self, image_path):
        """
        Extract text with layout information (useful for UI text).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: OCR result with layout information
        """
        try:
            self._load_reader()
            
            image = Image.open(image_path)
            results = self.reader.readtext(np.array(image))
            
            layout_results = []
            
            for (bbox, text, confidence) in results:
                text = text.strip()
                if text:
                    x_center = (bbox[0][0] + bbox[2][0]) / 2
                    y_center = (bbox[0][1] + bbox[2][1]) / 2
                    
                    layout_results.append({
                        'text': text,
                        'confidence': round(confidence, 4),
                        'bbox': bbox,
                        'position': {'x': round(x_center, 2), 'y': round(y_center, 2)},
                        'is_ui_element': self._is_likely_ui_text(text)
                    })
            
            layout_results.sort(key=lambda x: (x['position']['y'], x['position']['x']))
            
            ui_texts = [r['text'] for r in layout_results if r['is_ui_element']]
            
            return {
                'success': True,
                'all_texts': [r['text'] for r in layout_results],
                'ui_texts': ui_texts,
                'layout_results': layout_results,
                'language': self.languages
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'all_texts': [],
                'ui_texts': []
            }
    
    def _is_likely_ui_text(self, text):
        """
        Determine if text is likely a UI element.
        """
        text_lower = text.lower()
        
        ui_indicators = [
            'submit', 'cancel', 'ok', 'yes', 'no', 'save', 'delete', 'edit',
            'click', 'close', 'open', 'search', 'login', 'sign up', 'register',
            'menu', 'home', 'back', 'next', 'previous', 'continue',
            'http', 'www', '.com', '.org',
            'name', 'email', 'password', 'address', 'phone'
        ]
        
        is_short = len(text) <= 20
        has_indicator = any(indicator in text_lower for indicator in ui_indicators)
        
        return is_short or has_indicator
    
    def cleanup(self):
        """Clean up OCR resources."""
        self.reader = None


class TesseractOCRService:
    """
    Alternative OCR service using Tesseract.
    """
    
    def __init__(self, language='eng'):
        self.language = language
    
    def extract_text(self, image_path, detail_level='normal'):
        try:
            import pytesseract
            
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.language)
            all_text = [line.strip() for line in text.split('\n') if line.strip()]
            
            return {
                'success': True,
                'text': text.strip(),
                'all_text': all_text,
                'detailed_results': [],
                'language': self.language,
                'text_count': len(all_text)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': None,
                'all_text': []
            }
    
    def extract_text_from_image(self, image, detail_level='normal'):
        try:
            import pytesseract
            
            text = pytesseract.image_to_string(image, lang=self.language)
            all_text = [line.strip() for line in text.split('\n') if line.strip()]
            
            return {
                'success': True,
                'text': text.strip(),
                'all_text': all_text,
                'detailed_results': [],
                'language': self.language,
                'text_count': len(all_text)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': None,
                'all_text': []
            }


def get_ocr_service(provider="easyocr", languages=['en'], use_gpu=False):
    """
    Factory function to get the appropriate OCR service.
    """
    if provider.lower() == "tesseract":
        lang_code = '+'.join(languages) if isinstance(languages, list) else languages
        return TesseractOCRService(language=lang_code)
    else:
        return OCRService(languages=languages, use_gpu=use_gpu)
