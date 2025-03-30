#!/usr/bin/env python3
import argparse
import os
from src.web_interface import create_app

def main():
    """Run the Hindi-English Translator web interface."""
    parser = argparse.ArgumentParser(description="Hindi-English Translator Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", "-d", action="store_true", help="Run in debug mode")
    parser.add_argument("--models-dir", "-m", help="Directory containing fine-tuned models")
    
    args = parser.parse_args()
    
    # Create and run the Flask app
    app = create_app(custom_model_dir=args.models_dir)
    
    print(f"\nStarting Hindi-English Translator web interface on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 