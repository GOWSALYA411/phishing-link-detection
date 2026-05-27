from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import os

app = Flask(__name__)

# ============ TRUSTED DOMAINS WHITELIST ============
TRUSTED_DOMAINS = [
    # Search Engines
    'google.com', 'bing.com', 'yahoo.com', 'duckduckgo.com',
    
    # Social Media
    'facebook.com', 'instagram.com', 'twitter.com', 'linkedin.com',
    'whatsapp.com', 't.me', 'discord.com', 'reddit.com',
    
    # Development Platforms
    'github.com', 'gitlab.com', 'bitbucket.org', 'stackoverflow.com',
    'stackexchange.com', 'medium.com', 'dev.to', 'npmjs.com',
    
    # Data Science & ML Platforms
    'kaggle.com', 'kaggleusercontent.com', 'colab.research.google.com',
    'huggingface.co', 'paperswithcode.com', 'arxiv.org',
    'pypi.org', 'anaconda.org', 'jupyter.org',
    
    # Microsoft
    'microsoft.com', 'azure.com', 'visualstudio.com', 'windows.com',
    
    # Tech Companies
    'apple.com', 'amazon.com', 'netflix.com', 'spotify.com',
    'adobe.com', 'oracle.com', 'ibm.com', 'salesforce.com',
    
    # Programming Languages
    'python.org', 'nodejs.org', 'react.dev', 'angular.io',
    'vuejs.org', 'jquery.com', 'php.net', 'mysql.com',
    
    # Education
    'wikipedia.org', 'coursera.org', 'udemy.com', 'khanacademy.org',
    'mit.edu', 'stanford.edu', 'harvard.edu', 'oxford.ac.uk',
    
    # Government (Safe)
    '.gov', '.mil', '.edu', '.ac.uk', '.gov.in', '.gov.sg',
    
    # Popular Services
    'dropbox.com', 'drive.google.com', 'docs.google.com',
    'zoom.us', 'slack.com', 'teams.microsoft.com',
    
    # AI & Tech Companies
    'deepseek.com', 'deepseek.ai',
    'openai.com', 'chat.openai.com',
    'anthropic.com', 'claude.ai',
    'perplexity.ai', 'perplexity.com',
]

# ============ TRUSTED DOMAIN CHECK FUNCTION ============
def is_trusted_domain(url):
    """Check if URL belongs to trusted domain"""
    url_lower = url.lower()
    
    # Direct domain match from TRUSTED_DOMAINS list
    for domain in TRUSTED_DOMAINS:
        if domain in url_lower:
            return True
    
    # TLD-based trust
    trusted_tlds = ['.gov', '.mil', '.edu', '.ac.uk', '.gov.in', '.gov.sg']
    for tld in trusted_tlds:
        if tld in url_lower:
            return True
    
    return False

# ============ 🔴 NEW: TRUSTED VS HIGH-RISK SHORTENERS ============
TRUSTED_SHORTENERS = [
    'goo.gl',      # Google - Safe
    'ow.ly',       # Hootsuite - Safe
    'buff.ly',     # Buffer - Safe
    'is.gd',       # is.gd - Generally Safe
]

HIGH_RISK_SHORTENERS = [
    'bit.ly',      # Most abused phishing shortener
    'tinyurl.com', # Second most abused
    'tiny.cc',     # Commonly abused
    'cutt.ly',     # Phishing campaigns
    'shorturl.at', # Phishing campaigns
    'rebrandly',   # Often abused
    't.co',        # Twitter/X - but often abused
]

def classify_shortener(url):
    """Classify URL shorteners as Safe or High-Risk"""
    url_lower = url.lower()
    
    # Check for trusted safe shorteners FIRST
    for shortener in TRUSTED_SHORTENERS:
        if shortener in url_lower:
            return 'safe', shortener.upper()
    
    # Then check for high-risk shorteners
    for shortener in HIGH_RISK_SHORTENERS:
        if shortener in url_lower:
            return 'high_risk', shortener.upper()
    
    return 'unknown', None

# ============ LOAD MODEL ============
try:
    model = joblib.load('models/phishing_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/model_features.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    print("✅" + "="*50)
    print("✅ MODEL LOADED SUCCESSFULLY!")
    print("✅" + "="*50)
    print(f"📊 Features: {len(feature_names)}")
    print(f"📋 Classes: {encoder.classes_}")
    model_loaded = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_loaded = False
    feature_names = []

def extract_features(url):
    """Extract features from URL"""
    features = {}
    
    try:
        # Parse URL
        parsed = urlparse(url)
        netloc = parsed.netloc or parsed.path
        
        # Basic features
        features['url_length'] = len(url)
        features['hostname_length'] = len(netloc)
        features['ip'] = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0
        features['total_of.'] = url.count('.')
        features['total_of-'] = url.count('-')
        features['total_of/'] = url.count('/')
        features['total_of?'] = url.count('?')
        features['total_of='] = url.count('=')
        features['total_of@'] = url.count('@')
        features['total_of_'] = url.count('_')
        features['https_token'] = 1 if parsed.scheme == 'https' else 0
        
        # Digits
        digits = re.findall(r'\d', url)
        features['ratio_digits_url'] = len(digits) / len(url) if len(url) > 0 else 0
        
        # Subdomains
        domain_parts = netloc.replace('www.', '').split('.')
        features['nb_subdomains'] = max(0, len(domain_parts) - 2)
        
        # Suspicious words
        suspicious = ['secure', 'account', 'login', 'signin', 'verify', 'bank', 
                      'paypal', 'update', 'confirm', 'wallet', 'password', 'credential']
        features['phish_hints'] = sum(1 for word in suspicious if word in url.lower())
        
        # URL shorteners - Note: We're not using this for decision anymore
        # But keeping for model compatibility
        all_shorteners = TRUSTED_SHORTENERS + HIGH_RISK_SHORTENERS
        features['shortening_service'] = 1 if any(s in url.lower() for s in all_shorteners) else 0
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Add missing features
        if feature_names:
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_names]
        
        return df
    
    except Exception as e:
        print(f"Feature extraction error: {e}")
        if feature_names:
            return pd.DataFrame([[0]*len(feature_names)], columns=feature_names)
        return pd.DataFrame()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', 
                         model_loaded=model_loaded)

# ============ MAIN PREDICTION FUNCTION ============
@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return render_template('index.html', 
                             error="Model not loaded! Please train first.", 
                             model_loaded=False)
    
    url = request.form['url']
    
    try:
        # ============ 🔴 FIRST: CHECK SHORTENERS (MOST IMPORTANT) ============
        shortener_status, shortener_name = classify_shortener(url)
        
        # HIGH-RISK SHORTENERS - Immediate PHISHING
        if shortener_status == 'high_risk':
            return render_template('index.html',
                                 url=url,
                                 result='PHISHING',
                                 confidence='99.9%',
                                 message=f'⚠️ {shortener_name} - High Risk Shortener, commonly used for phishing!',
                                 alert_class='danger',
                                 icon='⚠️',
                                 model_loaded=model_loaded)
        
        # SAFE SHORTENERS - Immediate LEGITIMATE
        if shortener_status == 'safe':
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message=f'✅ {shortener_name} - Trusted URL shortener',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        # ============ SPECIAL CASES (Trusted Platforms) ============
        if 'kaggle.com' in url.lower():
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ KAGGLE - Trusted data science platform',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        if 'github.com' in url.lower():
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ GITHUB - Trusted development platform',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        if 'google.com' in url.lower():
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ GOOGLE - Trusted search engine',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        if 'python.org' in url.lower():
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ PYTHON.ORG - Official Python website',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        if 'microsoft.com' in url.lower() or 'azure.com' in url.lower():
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ MICROSOFT - Trusted technology company',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        if 'stackoverflow.com' in url.lower() or 'stackexchange.com' in url.lower():
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ STACK OVERFLOW - Trusted developer community',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        if 'amazon.com' in url.lower() and not any(x in url.lower() for x in ['verify', 'secure', 'update', 'confirm']):
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ AMAZON - Trusted e-commerce platform',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        if '.edu' in url.lower() or '.gov' in url.lower() or '.mil' in url.lower():
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ .EDU/.GOV - Trusted educational/government domain',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        if 'colab.research.google.com' in url.lower():
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ GOOGLE COLAB - Trusted ML platform',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        if 'huggingface.co' in url.lower():
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ HUGGING FACE - Trusted AI platform',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        if 'deepseek.com' in url.lower() or 'deepseek.ai' in url.lower():
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='99.9%',
                                 message='✅ DEEPSEEK - Trusted AI platform',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        # ============ TRUSTED DOMAIN CHECK ============
        if is_trusted_domain(url):
            return render_template('index.html',
                                 url=url,
                                 result='LEGITIMATE',
                                 confidence='95.0%',
                                 message='✅ TRUSTED DOMAIN - This website is legitimate',
                                 alert_class='success',
                                 icon='✅',
                                 model_loaded=model_loaded)
        
        # ============ MODEL PREDICTION FOR UNKNOWN DOMAINS ============
        # Extract features
        features_df = extract_features(url)
        
        if features_df.empty:
            raise Exception("Feature extraction failed")
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Predict
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        
        # Get result
        if encoder:
            result = encoder.inverse_transform([pred])[0]
        else:
            result = "PHISHING" if pred == 1 else "LEGITIMATE"
        
        confidence = max(proba) * 100
        
        # Determine risk
        if result.lower() == 'phishing' and confidence > 60:
            alert_class = 'danger'
            icon = '⚠️'
            message = '⚠️ PHISHING DETECTED! Do not enter personal information!'
        else:
            result = 'LEGITIMATE'
            alert_class = 'success'
            icon = '✅'
            message = '✅ This website appears legitimate'
        
        return render_template('index.html',
                             url=url,
                             result=result,
                             confidence=f"{confidence:.1f}%",
                             message=message,
                             alert_class=alert_class,
                             icon=icon,
                             model_loaded=model_loaded)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html',
                             url=url,
                             error=f"Error analyzing URL: {str(e)}",
                             model_loaded=model_loaded)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)