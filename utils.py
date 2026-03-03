"""
Utility functions for the Visual Localization Module.
"""
import os
import json
from datetime import datetime
from flask import jsonify

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Feedback storage file
FEEDBACK_FILE = 'feedback.json'


def allowed_file(filename):
    """
    Check if the file extension is allowed.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file(file, max_size=MAX_FILE_SIZE):
    """
    Validate uploaded file.
    Returns (is_valid, error_message)
    """
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, f"File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}"
    
    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > max_size:
        return False, f"File too large. Maximum size: {max_size // (1024*1024)}MB"
    
    return True, None


def create_response(success, data=None, error=None, status_code=200):
    """
    Create standardized JSON response.
    """
    if success:
        response = {
            'success': True,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
    else:
        response = {
            'success': False,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    return jsonify(response), status_code


def save_feedback(feedback_data):
    """
    Save user feedback to JSON file.
    """
    feedback_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'feedback': feedback_data.get('feedback'),
        'localized_caption': feedback_data.get('localized_caption'),
        'language': feedback_data.get('language'),
        'region': feedback_data.get('region'),
        'user_comment': feedback_data.get('comment', '')
    }
    
    # Load existing feedback
    feedbacks = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                feedbacks = json.load(f)
        except (json.JSONDecodeError, IOError):
            feedbacks = []
    
    # Add new feedback
    feedbacks.append(feedback_entry)
    
    # Save to file
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedbacks, f, indent=2)
    
    return True


def load_feedback():
    """
    Load all feedback from JSON file.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return []
    
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def get_config():
    """
    Get configuration settings.
    """
    return {
        'max_file_size': MAX_FILE_SIZE,
        'allowed_extensions': list(ALLOWED_EXTENSIONS),
        'supported_languages': [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ko', 'ar',
            'hi', 'ru', 'nl', 'pl', 'tr', 'vi', 'th', 'id', 'ms', 'sv'
        ],
        'supported_tones': ['formal', 'casual', 'professional', 'friendly'],
        'supported_regions': [
            'us', 'uk', 'ca', 'au', 'in', 'br', 'mx', 'fr', 'de', 'es',
            'it', 'jp', 'cn', 'kr', 'sa', 'ae', 'za', 'ng', 'ke', 'eg'
        ]
    }
