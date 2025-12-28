#!/usr/bin/env python3
"""
Web Server Runner for Biometric Authentication System
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.app import app

if __name__ == '__main__':
    print("=" * 60)
    print("🌐 BIOMETRIC AUTHENTICATION WEB SERVER")
    print("=" * 60)
    print("")
    print("📍 Server URL: http://127.0.0.1:5000")
    print("📍 Dashboard:  http://127.0.0.1:5000/dashboard")
    print("📍 Enroll:     http://127.0.0.1:5000/enroll")
    print("📍 Auth:       http://127.0.0.1:5000/authenticate")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print("")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )
