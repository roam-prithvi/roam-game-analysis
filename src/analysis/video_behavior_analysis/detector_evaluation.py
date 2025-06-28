"""
Detector evaluation script for comparing YOLO vs Grounding DINO + SAM performance.

This script provides comprehensive evaluation metrics and visual comparisons
between different detection methods on game footage.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .base_detector import BaseDetector, Detection
from .grounded_sam_detector import create_grounded_sam_detector
from .yolo_detector import create_yolo_detector
from .game_prompts import get_game_prompts, get_primary_game_prompts


class DetectorEvaluator:
    """Comprehensive evaluation of different detectors on game footage."""
    
    def __init__(self, test_images_dir: Path, game_name: str):
        """
        Initialize evaluator.
        
        Args:
            test_images_dir: Directory containing test images
            game_name: Name of the game for context-specific evaluation
        """
        self.test_images_dir = test_images_dir
        self.game_name = game_name
        self.test_images = list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg"))
        
        # Game-specific prompts
        self.game_prompts = get_game_prompts(game_name)
        self.primary_prompts = get_primary_game_prompts(game_name)
        
        print(f"🎮 Evaluating on {game_name} with {len(self.test_images)} test images")
        print(f"🎯 Using {len(self.game_prompts)} game-specific prompts")
    
    async def evaluate_detector(self, 
                               detector: BaseDetector, 
                               detector_name: str) -> Dict[str, Any]:
        """Evaluate a single detector on all test images."""
        print(f"\n🔧 Evaluating {detector_name} detector...")
        
        start_time = time.time()
        all_detections = []
        processing_times = []
        
        for i, image_path in enumerate(self.test_images):
            print(f"📸 Processing image {i+1}/{len(self.test_images)}: {image_path.name}")
            
            # Time detection
            detection_start = time.time()
            
            # Use appropriate prompts based on detector capabilities
            if detector.get_supported_features().get('text_prompts', False):
                detections = await detector.detect(str(image_path), prompts=self.primary_prompts)
            else:
                detections = await detector.detect(str(image_path), prompts=self.game_prompts)
            
            detection_time = time.time() - detection_start
            processing_times.append(detection_time)
            
            # Store results
            image_result = {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'detections': detections,
                'processing_time': detection_time,
                'detection_count': len(detections)
            }
            all_detections.append(image_result)
            
            print(f"   ✅ Found {len(detections)} objects in {detection_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_detections, processing_times, total_time)
        
        return {
            'detector_name': detector_name,
            'metrics': metrics,
            'detailed_results': all_detections,
            'features': detector.get_supported_features()
        }
    
    def _calculate_metrics(self, 
                          results: List[Dict], 
                          processing_times: List[float], 
                          total_time: float) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        total_detections = sum(r['detection_count'] for r in results)
        
        # Performance metrics
        avg_processing_time = np.mean(processing_times)
        std_processing_time = np.std(processing_times)
        fps = len(self.test_images) / total_time
        
        # Detection metrics
        avg_detections_per_image = total_detections / len(results) if results else 0
        
        # Confidence distribution
        all_confidences = []
        class_distribution = {}
        
        for result in results:
            for det in result['detections']:
                all_confidences.append(det.confidence)
                class_name = det.class_name
                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        
        confidence_stats = {
            'mean': np.mean(all_confidences) if all_confidences else 0,
            'std': np.std(all_confidences) if all_confidences else 0,
            'min': np.min(all_confidences) if all_confidences else 0,
            'max': np.max(all_confidences) if all_confidences else 0
        }
        
        return {
            'total_images': len(results),
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections_per_image,
            'avg_processing_time': avg_processing_time,
            'std_processing_time': std_processing_time,
            'fps': fps,
            'confidence_stats': confidence_stats,
            'class_distribution': class_distribution,
            'processing_times': processing_times
        }
    
    def create_visual_comparison(self, 
                               results: Dict[str, Dict], 
                               output_dir: Path, 
                               sample_images: int = 5):
        """Create visual comparison images showing detection results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select sample images for comparison
        sample_indices = np.linspace(0, len(self.test_images) - 1, sample_images, dtype=int)
        
        for idx in sample_indices:
            image_path = self.test_images[idx]
            image_name = image_path.stem
            
            # Load original image
            original = Image.open(image_path).convert("RGB")
            
            # Create comparison figure
            num_detectors = len(results)
            fig, axes = plt.subplots(1, num_detectors + 1, figsize=(5 * (num_detectors + 1), 5))
            
            # Show original image
            axes[0].imshow(original)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            # Show results from each detector
            for i, (detector_name, result) in enumerate(results.items()):
                image_result = result['detailed_results'][idx]
                detections = image_result['detections']
                
                # Draw bounding boxes
                vis_image = original.copy()
                draw = ImageDraw.Draw(vis_image)
                
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
                
                for j, det in enumerate(detections):
                    color = colors[j % len(colors)]
                    bbox = det.bbox
                    
                    # Draw bounding box
                    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], 
                                 outline=color, width=2)
                    
                    # Draw label
                    label = f"{det.class_name} ({det.confidence:.2f})"
                    draw.text((bbox[0], bbox[1] - 15), label, fill=color)
                
                axes[i + 1].imshow(vis_image)
                axes[i + 1].set_title(f"{detector_name}\n{len(detections)} objects")
                axes[i + 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"comparison_{image_name}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Saved comparison for {image_name}")
    
    def create_performance_plots(self, results: Dict[str, Dict], output_dir: Path):
        """Create performance comparison plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        detector_names = list(results.keys())
        
        # Processing time comparison
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Average processing time
        plt.subplot(2, 2, 1)
        avg_times = [results[name]['metrics']['avg_processing_time'] for name in detector_names]
        std_times = [results[name]['metrics']['std_processing_time'] for name in detector_names]
        
        plt.bar(detector_names, avg_times, yerr=std_times, capsize=5)
        plt.title('Average Processing Time per Image')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        # Plot 2: Detections per image
        plt.subplot(2, 2, 2)
        avg_detections = [results[name]['metrics']['avg_detections_per_image'] for name in detector_names]
        
        plt.bar(detector_names, avg_detections)
        plt.title('Average Detections per Image')
        plt.ylabel('Number of Objects')
        plt.xticks(rotation=45)
        
        # Plot 3: FPS comparison
        plt.subplot(2, 2, 3)
        fps_values = [results[name]['metrics']['fps'] for name in detector_names]
        
        plt.bar(detector_names, fps_values)
        plt.title('Processing Speed (FPS)')
        plt.ylabel('Frames per Second')
        plt.xticks(rotation=45)
        
        # Plot 4: Confidence distribution
        plt.subplot(2, 2, 4)
        for name in detector_names:
            confidences = results[name]['metrics']['confidence_stats']
            plt.bar(name, confidences['mean'], 
                   yerr=confidences['std'], 
                   alpha=0.7, 
                   label=f"{name} (σ={confidences['std']:.2f})")
        
        plt.title('Average Detection Confidence')
        plt.ylabel('Confidence Score')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("📈 Saved performance comparison plots")
    
    def generate_report(self, results: Dict[str, Dict], output_file: Path):
        """Generate comprehensive evaluation report."""
        report = {
            'evaluation_summary': {
                'game': self.game_name,
                'test_images': len(self.test_images),
                'game_prompts': self.game_prompts,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'detector_results': results,
            'comparison': {}
        }
        
        # Add comparison insights
        detector_names = list(results.keys())
        if len(detector_names) >= 2:
            # Compare processing speed
            fastest = min(detector_names, key=lambda x: results[x]['metrics']['avg_processing_time'])
            slowest = max(detector_names, key=lambda x: results[x]['metrics']['avg_processing_time'])
            
            # Compare detection count
            most_detections = max(detector_names, key=lambda x: results[x]['metrics']['avg_detections_per_image'])
            least_detections = min(detector_names, key=lambda x: results[x]['metrics']['avg_detections_per_image'])
            
            report['comparison'] = {
                'fastest_detector': fastest,
                'slowest_detector': slowest,
                'speed_ratio': results[slowest]['metrics']['avg_processing_time'] / results[fastest]['metrics']['avg_processing_time'],
                'most_detections': most_detections,
                'least_detections': least_detections,
                'detection_ratio': results[most_detections]['metrics']['avg_detections_per_image'] / max(results[least_detections]['metrics']['avg_detections_per_image'], 1)
            }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📋 Evaluation report saved to {output_file}")
        
        # Print summary
        print(f"\n📊 Evaluation Summary for {self.game_name}:")
        print(f"📸 Test images: {len(self.test_images)}")
        
        for name, result in results.items():
            metrics = result['metrics']
            print(f"\n🔧 {name}:")
            print(f"   ⏱️  Avg time: {metrics['avg_processing_time']:.3f}s")
            print(f"   🎯 Avg detections: {metrics['avg_detections_per_image']:.1f}")
            print(f"   📈 FPS: {metrics['fps']:.1f}")
            print(f"   🎲 Confidence: {metrics['confidence_stats']['mean']:.3f} ± {metrics['confidence_stats']['std']:.3f}")
    
    async def run_comparison(self, output_dir: Path):
        """Run full comparison between YOLO and Grounding DINO + SAM."""
        print(f"🔬 Starting comprehensive detector evaluation...")
        print(f"🎮 Game: {self.game_name}")
        print(f"📁 Output: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detectors
        print("\n🔧 Initializing detectors...")
        yolo_detector = await create_yolo_detector(confidence_threshold=0.3)
        grounded_detector = await create_grounded_sam_detector(confidence_threshold=0.3)
        
        # Evaluate both detectors
        results = {}
        
        # Evaluate YOLO
        yolo_results = await self.evaluate_detector(yolo_detector, "YOLO + SAM")
        results["YOLO + SAM"] = yolo_results
        
        # Evaluate Grounding DINO + SAM
        grounded_results = await self.evaluate_detector(grounded_detector, "Grounding DINO + SAM")
        results["Grounding DINO + SAM"] = grounded_results
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        self.create_visual_comparison(results, output_dir / "visual_comparisons")
        self.create_performance_plots(results, output_dir)
        
        # Generate report
        self.generate_report(results, output_dir / "evaluation_report.json")
        
        print(f"\n✅ Evaluation complete! Results saved to {output_dir}")
        
        return results


async def main():
    """CLI entry point for detector evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate detectors on game footage")
    parser.add_argument("test_images_dir", help="Directory containing test images")
    parser.add_argument("--game", required=True, help="Game name for context")
    parser.add_argument("--output", default="./evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_images_dir)
    if not test_dir.exists():
        print(f"❌ Test images directory not found: {test_dir}")
        return
    
    output_dir = Path(args.output)
    
    evaluator = DetectorEvaluator(test_dir, args.game)
    await evaluator.run_comparison(output_dir)


if __name__ == "__main__":
    asyncio.run(main()) 