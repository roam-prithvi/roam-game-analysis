"""
Universal object extractor for game analysis.

This module provides game-agnostic object detection and segmentation using:
1. Universal prompts that work for any game
2. Gemini-driven object discovery from video analysis
3. Dynamic object detection without pre-defined game labels
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Universal prompts that work across all games
UNIVERSAL_OBJECT_PROMPTS = [
    # Characters and entities
    "character", "person", "player", "hero", "avatar",
    "enemy", "opponent", "monster", "creature", "animal",
    "robot", "vehicle", "car", "truck", "ship", "plane",
    
    # Interactive objects
    "button", "lever", "switch", "door", "platform",
    "obstacle", "barrier", "wall", "block", "box",
    "collectible", "item", "object", "tool", "weapon",
    "coin", "gem", "star", "heart", "key", "powerup",
    
    # Environment
    "building", "house", "tree", "rock", "water", "fire",
    "ground", "floor", "ceiling", "bridge", "stairs",
    "tower", "structure", "decoration", "sign", "text",
    
    # UI elements
    "icon", "symbol", "meter", "bar", "counter", "score",
    "health", "energy", "progress", "timer", "map"
]

# Generic category mappings
CATEGORY_KEYWORDS = {
    'player': ['character', 'player', 'hero', 'avatar', 'person running', 'main character'],
    'enemies': ['enemy', 'opponent', 'monster', 'zombie', 'alien', 'guard', 'robot enemy'],
    'obstacles': ['obstacle', 'barrier', 'wall', 'block', 'train', 'car', 'truck', 'spike'],
    'collectibles': ['coin', 'gem', 'star', 'heart', 'powerup', 'collectible', 'item', 'bonus'],
    'ui': ['button', 'icon', 'meter', 'bar', 'score', 'health', 'timer', 'menu'],
    'environment': ['building', 'tree', 'platform', 'ground', 'water', 'background', 'decoration'],
    'interactive': ['door', 'lever', 'switch', 'tool', 'weapon', 'key', 'portal']
}


class UniversalObjectExtractor:
    """Universal object extractor that works with any game."""
    
    def __init__(self, session_path: Union[str, Path], detector_type: str = "grounded_sam"):
        """
        Initialize the universal extractor.
        
        Args:
            session_path: Path to session directory
            detector_type: Type of detector to use
        """
        self.session_path = Path(session_path)
        self.detector_type = detector_type
        self.analysis_dir = self.session_path / "analysis" / "universal_objects"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Detected objects from Gemini analysis
        self.gemini_objects: Set[str] = set()
        
        # All discovered objects across frames
        self.discovered_objects: Set[str] = set()
    
    def extract_objects_from_gemini_analysis(self, analysis_text: str) -> Set[str]:
        """
        Extract object names from Gemini's video analysis text.
        
        Args:
            analysis_text: Raw text output from Gemini video analysis
            
        Returns:
            Set of object names found in the analysis
        """
        objects = set()
        
        # Look for object specifications in Gemini's structured analysis
        object_patterns = [
            # Look for "Object Name:" patterns
            r'(?:Object Name|Entity|Character|Item):\s*([^\n\r]+)',
            # Look for quoted object names
            r'"([^"]+)"',
            # Look for behavioral specifications
            r'(?:moving|static|animated|visible)\s+([a-zA-Z][a-zA-Z\s]+?)(?:\s+(?:that|which|with|moves|appears))',
            # Look for "X does Y" patterns
            r'([A-Z][a-zA-Z\s]+?)\s+(?:moves|attacks|collects|appears|spawns|shoots)',
            # Look for lists of objects
            r'(?:objects|entities|items|characters).*?:\s*([^\n\r.]+)',
        ]
        
        for pattern in object_patterns:
            matches = re.finditer(pattern, analysis_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                obj_text = match.group(1).strip()
                
                # Clean and extract individual objects
                obj_names = self._clean_object_names(obj_text)
                objects.update(obj_names)
        
        # Also look for section headers that might contain object names
        section_patterns = [
            r'\*\*(.*?)\*\*',  # **Object Name**
            r'(?:SECTION|OBJECT|ENTITY|CHARACTER)\s+\d+:\s*([^\n\r]+)',
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, analysis_text, re.IGNORECASE)
            for match in matches:
                obj_text = match.group(1).strip()
                obj_names = self._clean_object_names(obj_text)
                objects.update(obj_names)
        
        # Filter out generic terms and keep meaningful object names
        filtered_objects = set()
        generic_terms = {
            'object', 'entity', 'character', 'item', 'element', 'component',
            'thing', 'part', 'section', 'area', 'zone', 'region', 'space',
            'behavior', 'movement', 'action', 'interaction', 'response'
        }
        
        for obj in objects:
            obj_lower = obj.lower()
            if (len(obj) > 2 and 
                obj_lower not in generic_terms and
                not obj_lower.startswith(('the ', 'a ', 'an ')) and
                any(c.isalpha() for c in obj)):
                filtered_objects.add(obj)
        
        self.gemini_objects.update(filtered_objects)
        print(f"🧠 Extracted {len(filtered_objects)} objects from Gemini analysis")
        
        return filtered_objects
    
    def _clean_object_names(self, text: str) -> List[str]:
        """Clean and split object names from text."""
        # Remove common prefixes/suffixes
        text = re.sub(r'(?:Complete |Behavioral |Visual )', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?:\s+Specification|\s+Analysis|\s+Description)$', '', text, flags=re.IGNORECASE)
        
        # Split on common separators
        objects = []
        separators = [',', ';', '&', ' and ', ' or ', '\n', '/', '|']
        
        parts = [text]
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts
        
        for part in parts:
            cleaned = part.strip(' \t\n\r"\'()[]{}')
            if len(cleaned) > 2 and len(cleaned) < 50:  # Reasonable object name length
                objects.append(cleaned)
        
        return objects
    
    def create_universal_prompt(self) -> str:
        """Create a universal Grounding DINO prompt."""
        # Combine universal prompts with Gemini-discovered objects
        all_prompts = set(UNIVERSAL_OBJECT_PROMPTS)
        all_prompts.update(self.gemini_objects)
        all_prompts.update(self.discovered_objects)
        
        # Limit to most relevant prompts to avoid overwhelming the model
        prompt_list = list(all_prompts)[:30]  # Limit to 30 most diverse prompts
        
        return " . ".join(prompt_list) + " ."
    
    def create_open_vocabulary_prompt(self) -> str:
        """Create an open-vocabulary prompt that detects anything."""
        return "object . item . thing . entity . character . element ."
    
    def categorize_universal_detection(self, class_name: str, bbox: List[float], 
                                    image_shape: Tuple[int, int]) -> str:
        """Categorize a detection using universal heuristics."""
        class_lower = class_name.lower()
        
        # Check against category keywords
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in class_lower for keyword in keywords):
                return category
        
        # Use positional and size heuristics
        x1, y1, x2, y2 = bbox
        img_height, img_width = image_shape
        
        # Size-based categorization
        width = x2 - x1
        height = y2 - y1
        area = width * height
        rel_area = area / (img_width * img_height)
        
        # Position-based categorization
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        rel_x = center_x / img_width
        rel_y = center_y / img_height
        
        # UI elements are usually small and at edges
        if rel_area < 0.01 and (rel_x < 0.1 or rel_x > 0.9 or rel_y < 0.1 or rel_y > 0.9):
            return 'ui'
        
        # Large objects covering significant area are likely environment
        if rel_area > 0.3:
            return 'environment'
        
        # Medium objects in the center might be player or enemies
        if 0.3 < rel_x < 0.7 and 0.2 < rel_y < 0.8:
            if 0.01 < rel_area < 0.1:
                return 'player'  # Main character is usually prominent but not huge
        
        # Small objects might be collectibles
        if 0.001 < rel_area < 0.02:
            return 'collectibles'
        
        return 'unknown'
    
    async def extract_frame_objects(self, frame_path: str, detector, frame_idx: int) -> Dict[str, Any]:
        """Extract all objects from a single frame using universal detection."""
        frame_name = Path(frame_path).stem
        print(f"🔍 Universally analyzing frame: {frame_name}")
        
        # Try multiple detection strategies
        all_detections = []
        
        # Strategy 1: Use universal prompts
        universal_prompt = self.create_universal_prompt()
        try:
            detections1 = await detector.detect(frame_path, prompts=[universal_prompt])
            all_detections.extend(detections1)
            print(f"  📝 Universal prompts found {len(detections1)} objects")
        except Exception as e:
            print(f"  ⚠️ Universal prompt detection failed: {e}")
        
        # Strategy 2: Use open vocabulary detection
        open_prompt = self.create_open_vocabulary_prompt()
        try:
            detections2 = await detector.detect(frame_path, prompts=[open_prompt])
            all_detections.extend(detections2)
            print(f"  🌐 Open vocabulary found {len(detections2)} objects")
        except Exception as e:
            print(f"  ⚠️ Open vocabulary detection failed: {e}")
        
        # Strategy 3: Use Gemini-discovered objects if available
        if self.gemini_objects:
            gemini_prompt = " . ".join(list(self.gemini_objects)[:20]) + " ."
            try:
                detections3 = await detector.detect(frame_path, prompts=[gemini_prompt])
                all_detections.extend(detections3)
                print(f"  🧠 Gemini objects found {len(detections3)} objects")
            except Exception as e:
                print(f"  ⚠️ Gemini object detection failed: {e}")
        
        # Remove duplicates and merge similar detections
        unique_detections = self._merge_duplicate_detections(all_detections)
        print(f"  🔗 Merged to {len(unique_detections)} unique objects")
        
        if not unique_detections:
            return {
                "frame_path": frame_path,
                "frame_idx": frame_idx,
                "objects_found": 0,
                "categories": {},
                "detections": []
            }
        
        # Categorize and save objects
        categorized_objects = {}
        detection_results = []
        
        # Load image to get dimensions
        image = cv2.imread(frame_path)
        img_height, img_width = image.shape[:2]
        
        for det in unique_detections:
            # Categorize detection
            category = self.categorize_universal_detection(
                det.class_name, det.bbox, (img_height, img_width)
            )
            
            if category not in categorized_objects:
                categorized_objects[category] = []
            categorized_objects[category].append(det)
            
            # Add to discovered objects
            self.discovered_objects.add(det.class_name)
            
            # Save object cutout
            cutout_path = self._save_object_cutout(
                frame_path, det, category, frame_idx, len(categorized_objects[category])
            )
            
            detection_results.append({
                "class_name": det.class_name,
                "category": category,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "cutout_path": cutout_path
            })
        
        category_counts = {cat: len(objects) for cat, objects in categorized_objects.items()}
        
        return {
            "frame_path": frame_path,
            "frame_idx": frame_idx,
            "objects_found": len(unique_detections),
            "categories": category_counts,
            "detections": detection_results
        }
    
    def _merge_duplicate_detections(self, detections, iou_threshold: float = 0.5):
        """Merge duplicate detections with high IoU overlap."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        merged = []
        for det in detections:
            should_merge = False
            
            for merged_det in merged:
                iou = self._calculate_iou(det.bbox, merged_det.bbox)
                if iou > iou_threshold:
                    # Keep the higher confidence detection
                    should_merge = True
                    break
            
            if not should_merge:
                merged.append(det)
        
        return merged
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _save_object_cutout(self, frame_path: str, detection, category: str, 
                          frame_idx: int, obj_idx: int) -> Optional[str]:
        """Save a cutout of the detected object."""
        try:
            frame_name = Path(frame_path).stem
            category_dir = self.analysis_dir / "objects" / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename
            output_name = f"{frame_name}_{category}_{detection.class_name}_{obj_idx}_{detection.confidence:.2f}.png"
            output_path = category_dir / output_name
            
            # Extract cutout
            if detection.mask is not None:
                success = self._extract_with_mask(frame_path, detection.mask, str(output_path))
            else:
                success = self._extract_with_bbox(frame_path, detection.bbox, str(output_path))
            
            if success:
                return str(output_path)
                
        except Exception as e:
            print(f"⚠️ Failed to save cutout: {e}")
        
        return None
    
    def _extract_with_mask(self, image_path: str, mask: np.ndarray, output_path: str) -> bool:
        """Extract object using segmentation mask."""
        try:
            # Load image
            image = Image.open(image_path).convert("RGBA")
            
            # Process mask
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            elif mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            
            # Create mask image
            mask_img = Image.fromarray(mask, mode='L')
            if mask_img.size != image.size:
                mask_img = mask_img.resize(image.size, Image.Resampling.LANCZOS)
            
            # Apply mask
            image_array = np.array(image)
            mask_array = np.array(mask_img)
            image_array[:, :, 3] = mask_array
            
            # Save
            cutout = Image.fromarray(image_array, "RGBA")
            cutout.save(output_path, "PNG")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Mask extraction failed: {e}")
            return False
    
    def _extract_with_bbox(self, image_path: str, bbox: List[float], output_path: str) -> bool:
        """Extract object using bounding box."""
        try:
            image = Image.open(image_path)
            x1, y1, x2, y2 = map(int, bbox)
            cropped = image.crop((x1, y1, x2, y2))
            
            if cropped.mode != "RGBA":
                cropped = cropped.convert("RGBA")
            
            cropped.save(output_path, "PNG")
            return True
            
        except Exception as e:
            print(f"⚠️ Bbox extraction failed: {e}")
            return False
    
    def save_analysis_summary(self, all_frame_results: List[Dict[str, Any]]) -> str:
        """Save a comprehensive analysis summary."""
        summary_path = self.analysis_dir / "universal_analysis_summary.json"
        
        # Compile statistics
        total_objects = sum(result['objects_found'] for result in all_frame_results)
        all_categories = set()
        all_object_types = set()
        
        for result in all_frame_results:
            all_categories.update(result['categories'].keys())
            for det in result.get('detections', []):
                all_object_types.add(det['class_name'])
        
        category_totals = {cat: 0 for cat in all_categories}
        for result in all_frame_results:
            for cat, count in result['categories'].items():
                category_totals[cat] += count
        
        summary = {
            "analysis_type": "universal_object_extraction",
            "detector_type": self.detector_type,
            "session_path": str(self.session_path),
            "frames_analyzed": len(all_frame_results),
            "total_objects_detected": total_objects,
            "unique_object_types": len(all_object_types),
            "categories_found": len(all_categories),
            "gemini_discovered_objects": list(self.gemini_objects),
            "all_discovered_objects": list(self.discovered_objects),
            "category_totals": category_totals,
            "object_types": list(all_object_types),
            "frame_results": all_frame_results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Analysis summary saved to: {summary_path}")
        return str(summary_path) 