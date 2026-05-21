"""
Image Captioning Service using OpenAI Vision API.
"""
import os
import base64

 
class CaptioningService:
    """
    Service for generating detailed image descriptions using OpenAI Vision API.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the captioning service.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
    
    def generate_caption(self, image_path, max_length=100):
        """
        Generate a detailed caption for the image.
        
        Args:
            image_path: Path to the image file
            max_length: Ignored for OpenAI
            
        Returns:
            dict: Caption result
        """
        try:
            from openai import OpenAI
            
            if not self.api_key:
                return {
                    'success': False,
                    'error': 'OpenAI API key not provided',
                    'caption': None
                }
            
            client = OpenAI(api_key=self.api_key)
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Provide a detailed description of this image including: objects visible, setting/environment, emotions/mood, and the general purpose or context. Keep it concise but descriptive."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            caption = response.choices[0].message.content
            
            return {
                'success': True,
                'caption': caption,
                'model': 'gpt-4o-mini',
                'description': caption
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'caption': None
            }
    
    def generate_caption_from_image(self, image, max_length=100):
        """
        Generate caption from PIL Image object.
        
        Args:
            image: PIL Image object
            max_length: Ignored for OpenAI
            
        Returns:
            dict: Caption result
        """
        try:
            from openai import OpenAI
            
            if not self.api_key:
                return {
                    'success': False,
                    'error': 'OpenAI API key not provided',
                    'caption': None
                }
            
            client = OpenAI(api_key=self.api_key)
            
            # Save to temp and read
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp.name)
                with open(tmp.name, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                os.unlink(tmp.name)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Provide a detailed description of this image including: objects visible, setting/environment, emotions/mood, and the general purpose or context. Keep it concise but descriptive."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            caption = response.choices[0].message.content
            
            return {
                'success': True,
                'caption': caption,
                'model': 'gpt-4o-mini',
                'description': caption
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'caption': None
            }
    
    def cleanup(self):
        """Clean up resources."""
        pass


def get_captioning_service(provider="openai", api_key=None):
    """
    Factory function to get the captioning service.
    
    Args:
        provider: 'openai' (currently only OpenAI is supported)
        api_key: API key for OpenAI
        
    Returns:
        CaptioningService instance
    """
    return CaptioningService(api_key=api_key)
