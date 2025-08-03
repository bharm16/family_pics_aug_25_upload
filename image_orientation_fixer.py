#!/usr/bin/env python3
"""
Image Orientation Fixer using OpenAI Vision API
Analyzes images to detect orientation and rotates them if needed.
"""

import os
import base64
from pathlib import Path
from PIL import Image, ExifTags
from openai import OpenAI
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

def setup_logging(log_file="image_processing.log"):
    """Setup dual logging to both console and file"""
    # Create logger
    logger = logging.getLogger('ImageProcessor')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (clean)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class ImageOrientationFixer:
    def __init__(self, api_key=None, max_workers=15, tracking_file="processed_images.json", log_file="image_processing.log"):
        """Initialize with OpenAI API key and concurrency settings"""
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.processed_count = 0
        self.rotated_count = 0
        self.tracking_file = tracking_file
        self.processed_images = self.load_tracking_data()
        self.logger = setup_logging(log_file)
    
    def load_tracking_data(self):
        """Load processed images tracking data"""
        try:
            if Path(self.tracking_file).exists():
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                    if hasattr(self, 'logger'):
                        self.logger.info(f"Loaded {len(data)} previously processed images from {self.tracking_file}")
                    return data
            return {}
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Could not load tracking file {self.tracking_file}: {e}")
            else:
                print(f"Warning: Could not load tracking file {self.tracking_file}: {e}")
            return {}
    
    def save_tracking_data(self):
        """Save processed images tracking data"""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.processed_images, f, indent=2)
            self.logger.debug(f"Saved tracking data for {len(self.processed_images)} images")
        except Exception as e:
            self.logger.error(f"Could not save tracking file {self.tracking_file}: {e}")
    
    def encode_image_to_base64(self, image_path):
        """Encode image to base64 for API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_image_orientation(self, image_path):
        """Use OpenAI Vision API to analyze image orientation"""
        file_name = Path(image_path).name
        self.logger.info(f"Analyzing {file_name}...")
        
        base64_image = self.encode_image_to_base64(image_path)
        
        prompt = """Analyze this image and determine if it needs to be rotated to be properly oriented. 

Look at any text, people, objects, or horizon lines to determine the correct orientation.

Respond with ONLY a JSON object in this exact format:
{
    "needs_rotation": true/false,
    "rotation_degrees": 0/90/180/270,
    "confidence": "high/medium/low",
    "reason": "brief explanation"
}

If the image is already correctly oriented, use needs_rotation: false and rotation_degrees: 0."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
                analysis = json.loads(json_text)
                
                # Log detailed analysis
                self.logger.info(f"  Analysis for {file_name}: {analysis}")
                return analysis
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            self.logger.error(f"Error analyzing {file_name}: {e}")
            return {"needs_rotation": False, "rotation_degrees": 0, "confidence": "low", "reason": "API error"}
    
    def rotate_image(self, image_path, degrees, output_path=None):
        """Rotate image by specified degrees"""
        file_name = Path(image_path).name
        
        if degrees == 0:
            return image_path
            
        try:
            with Image.open(image_path) as img:
                # Remove EXIF orientation data to avoid conflicts
                if hasattr(img, '_getexif'):
                    exif = img._getexif()
                    if exif is not None:
                        for tag, value in exif.items():
                            if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                                # Reset orientation to normal
                                exif[tag] = 1
                
                # Rotate the image
                rotated = img.rotate(-degrees, expand=True)
                
                # Save to output path or overwrite original
                save_path = output_path or image_path
                rotated.save(save_path, quality=95, optimize=True)
                
                self.logger.info(f"  Rotated {file_name} by {degrees} degrees -> {Path(save_path).name}")
                return save_path
                
        except Exception as e:
            self.logger.error(f"Error rotating {file_name}: {e}")
            return image_path
    
    def process_single_image(self, image_path, output_dir=None):
        """Process a single image (thread-safe version)"""
        try:
            file_name = Path(image_path).name
            
            analysis = self.analyze_image_orientation(image_path)
            
            was_rotated = False
            if analysis["needs_rotation"] and analysis["rotation_degrees"] > 0:
                if output_dir:
                    output_path = Path(output_dir) / file_name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = image_path
                    
                self.rotate_image(image_path, analysis["rotation_degrees"], str(output_path))
                was_rotated = True
            
            # Thread-safe progress and tracking update
            with self.progress_lock:
                self.processed_count += 1
                if was_rotated:
                    self.rotated_count += 1
                
                # Update tracking data
                tracking_data = {
                    'analyzed': True,
                    'rotated': was_rotated,
                    'rotation_degrees': analysis.get("rotation_degrees", 0),
                    'confidence': analysis.get("confidence", "unknown"),
                    'reason': analysis.get("reason", ""),
                    'timestamp': time.time()
                }
                self.processed_images[file_name] = tracking_data
                
                # Log what's being tracked
                status = "ROTATED" if was_rotated else "NO ROTATION"
                self.logger.info(f"  Tracked {file_name}: {status} ({analysis.get('confidence', 'unknown')} confidence)")
                
                # Log and save progress every 10 images
                if self.processed_count % 10 == 0:
                    progress_msg = f"Progress: {self.processed_count} processed, {self.rotated_count} rotated"
                    self.logger.info(progress_msg)
                    self.save_tracking_data()
            
            return {
                'file': file_name,
                'rotated': was_rotated,
                'analysis': analysis,
                'success': True
            }
            
        except Exception as e:
            file_name = Path(image_path).name
            # Still track failed attempts
            with self.progress_lock:
                error_data = {
                    'analyzed': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
                self.processed_images[file_name] = error_data
                self.logger.info(f"  Tracked {file_name}: FAILED (error tracked)")
            
            return {
                'file': file_name,
                'rotated': False,
                'error': str(e),
                'success': False
            }
    
    def process_directory(self, input_dir, output_dir=None, extensions=None):
        """Process all images in a directory with concurrent processing"""
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        input_path = Path(input_dir)
        
        # Reset counters
        self.processed_count = 0
        self.rotated_count = 0
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return
        
        # Check which files are already analyzed (from tracking file)
        already_analyzed = set(self.processed_images.keys())
        if already_analyzed:
            self.logger.info(f"Found {len(already_analyzed)} already analyzed images in tracking file:")
            analyzed_count = len([img for img in self.processed_images.values() if img.get('analyzed', False)])
            rotated_count = len([img for img in self.processed_images.values() if img.get('rotated', False)])
            failed_count = len([img for img in self.processed_images.values() if not img.get('analyzed', True)])
            self.logger.info(f"  - Successfully analyzed: {analyzed_count}")
            self.logger.info(f"  - Required rotation: {rotated_count}")
            self.logger.info(f"  - Failed analysis: {failed_count}")
        else:
            self.logger.info("No previously analyzed images found in tracking file")
        
        # Filter out already analyzed images
        images_to_process = [img for img in image_files if img.name not in already_analyzed]
        skipped_count = len(image_files) - len(images_to_process)
        
        self.logger.info(f"Found {len(image_files)} total images")
        self.logger.info(f"Will process {len(images_to_process)} new images using {self.max_workers} workers")
        
        if not images_to_process:
            self.logger.info("All images already processed!")
            return
        
        # Create output directory if it doesn't exist
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process images concurrently
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self.process_single_image, str(image_file), output_dir): image_file 
                for image_file in images_to_process
            }
            
            # Process completed futures
            for future in as_completed(future_to_image):
                result = future.result()
                results.append(result)
                
                if not result['success']:
                    self.logger.error(f"Error processing {result['file']}: {result.get('error', 'Unknown error')}")
        
        # Final save of tracking data
        self.save_tracking_data()
        
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"\nProcessing complete in {elapsed_time:.1f} seconds:")
        self.logger.info(f"  Total images: {len(image_files)}")
        self.logger.info(f"  Skipped (already analyzed): {skipped_count}")
        self.logger.info(f"  Newly processed: {self.processed_count}")
        self.logger.info(f"  Rotated: {self.rotated_count} images")
        if len(images_to_process) > 0:
            self.logger.info(f"  Average time per image: {elapsed_time/len(images_to_process):.1f}s")
            self.logger.info(f"  Processing rate: {len(images_to_process)*60/elapsed_time:.1f} images/minute")
        
        # Log any errors
        errors = [r for r in results if not r['success']]
        if errors:
            self.logger.error(f"  Errors: {len(errors)} images failed to process")
        
        # Show tracking file stats
        total_analyzed = len([img for img in self.processed_images.values() if img.get('analyzed', False)])
        total_rotated = len([img for img in self.processed_images.values() if img.get('rotated', False)])
        self.logger.info(f"  Total in tracking file: {len(self.processed_images)} images")
        self.logger.info(f"  Total successfully analyzed: {total_analyzed} images")
        self.logger.info(f"  Total rotated overall: {total_rotated} images")

def main():
    """Main function"""
    # Configuration - edit these values as needed
    api_key = "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"
    input_dir = "/Users/bryceharmon/Desktop/photos_ending_in_a copy"
    output_dir = "./corrected_images"
    max_workers = 15  # Optimal threading workers for good performance
    
    try:
        fixer = ImageOrientationFixer(api_key=api_key, max_workers=max_workers)
        
        fixer.logger.info("Starting Concurrent Image Orientation Fixer...")
        fixer.logger.info(f"Input directory: {input_dir}")
        fixer.logger.info(f"Output directory: {output_dir}")
        fixer.logger.info(f"Max concurrent workers: {max_workers}")
        fixer.logger.info(f"Log file: image_processing.log")
        fixer.logger.info("")
        
        input_path = Path(input_dir)
        
        if input_path.is_file():
            # Process single file
            result = fixer.process_single_image(str(input_path), output_dir)
            if result['success']:
                fixer.logger.info(f"Processed {result['file']}: {'rotated' if result['rotated'] else 'no rotation needed'}")
            else:
                fixer.logger.error(f"Error processing {result['file']}: {result.get('error', 'Unknown error')}")
        elif input_path.is_dir():
            # Process directory
            fixer.process_directory(str(input_path), output_dir)
        else:
            fixer.logger.error(f"Error: {input_path} is not a valid file or directory")
            return
            
    except ValueError as e:
        print(f"Error: {e}")
        return
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

if __name__ == "__main__":
    main()