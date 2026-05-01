"""
Flask Application for Visual Localization API.
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from visual_service import get_visual_localization_service
from utils import (   
    validate_file, create_response, save_feedback, 
    load_feedback, get_config
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize service (lazy loading)
visual_service = None


def get_service():
    """Get or create the visual localization service."""
    global visual_service
    if visual_service is None:
        config = {
            'captioning_provider': os.getenv('CAPTIONING_PROVIDER', 'blip'),
            'ocr_provider': os.getenv('OCR_PROVIDER', 'easyocr'),
            'localization_provider': os.getenv('LOCALIZATION_PROVIDER', 'openai'),
            'api_key': os.getenv('OPENAI_API_KEY'),
            'use_gpu': os.getenv('USE_GPU', 'false').lower() == 'true'
        }
        visual_service = get_visual_localization_service(config)
    return visual_service


@app.route('/')
def index():
    """Home endpoint."""
    return create_response(True, {
        'message': 'Visual Localization API',
        'version': '1.0.0',
        'endpoints': {
            'localize': '/api/localize (POST)',
            'feedback': '/api/feedback (POST)',
            'config': '/api/config (GET)'
        }
    })


@app.route('/api/config', methods=['GET'])
def config():
    """Get supported configuration options."""
    try:
        service = get_service()
        supported_config = service.get_supported_config()
        general_config = get_config()
        
        return create_response(True, {
            **supported_config,
            'max_file_size': general_config['max_file_size'],
            'allowed_extensions': general_config['allowed_extensions']
        })
    except Exception as e:
        return create_response(False, error=str(e), status_code=500)


@app.route('/api/localize', methods=['POST'])
def localize():
    """
    Main endpoint for visual localization.
    
    Expected form data:
    - image: Image file (required)
    - language: Target language code (required)
    - region: Target region code (required)
    - tone: Tone (formal/casual/professional/friendly) (required)
    - caption: Optional user caption (optional)
    - ocr_language: Language for OCR (optional, default: en)
    """
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return create_response(False, error='No image file provided', status_code=400)
        
        image_file = request.files['image']
        
        # Validate file
        is_valid, error_msg = validate_file(image_file)
        if not is_valid:
            return create_response(False, error=error_msg, status_code=400)
        
        # Get parameters
        target_language = request.form.get('language', 'en')
        target_region = request.form.get('region', 'us')
        tone = request.form.get('tone', 'friendly')
        user_caption = request.form.get('caption', None)
        ocr_language = request.form.get('ocr_language', 'en')
        
        # Validate required parameters
        if not target_language or not target_region or not tone:
            return create_response(
                False, 
                error='Missing required parameters: language, region, tone',
                status_code=400
            )
        
        # Process image
        service = get_service()
        result = service.process_image(
            image_file=image_file,
            target_language=target_language,
            target_region=target_region,
            tone=tone,
            user_caption=user_caption,
            ocr_language=ocr_language
        )
        
        if result['success']:
            return create_response(True, data=result)
        else:
            return create_response(False, error=result.get('error', 'Processing failed'), status_code=500)
    
    except Exception as e:
        return create_response(False, error=str(e), status_code=500)


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """
    Submit feedback for a localization result.
    
    Expected JSON:
    {
        "feedback": "helpful" or "not_helpful",
        "localized_caption": "...",
        "language": "...",
        "region": "...",
        "comment": "Optional comment"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return create_response(False, error='No JSON data provided', status_code=400)
        
        # Validate required fields
        required_fields = ['feedback', 'localized_caption', 'language', 'region']
        for field in required_fields:
            if field not in data:
                return create_response(
                    False, 
                    error=f'Missing required field: {field}',
                    status_code=400
                )
        
        # Validate feedback value
        if data['feedback'] not in ['helpful', 'not_helpful']:
            return create_response(
                False, 
                error='Invalid feedback value. Must be "helpful" or "not_helpful"',
                status_code=400
            )
        
        # Save feedback
        save_feedback(data)
        
        return create_response(True, data={'message': 'Feedback saved successfully'})
    
    except Exception as e:
        return create_response(False, error=str(e), status_code=500)


@app.route('/api/feedbacks', methods=['GET'])
def get_feedbacks():
    """Get all feedback entries."""
    try:
        feedbacks = load_feedback()
        return create_response(True, data={'feedbacks': feedbacks, 'count': len(feedbacks)})
    except Exception as e:
        return create_response(False, error=str(e), status_code=500)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return create_response(True, {'status': 'healthy'})


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return create_response(False, error='File too large. Maximum size is 10MB', status_code=413)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 error."""
    return create_response(False, error='Endpoint not found', status_code=404)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 error."""
    return create_response(False, error='Internal server error', status_code=500)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
