import cv2
import os
from tum_loader import TUMDataLoader
from object_pipeline import ObjectClassificationPipeline

def run_tum_demo(dataset_path, num_frames=50, output_dir='demo_output'):
    """Run demo on TUM RGB-D dataset"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load TUM dataset
    print("Loading TUM dataset...")
    loader = TUMDataLoader(dataset_path)
    print(f"Total frames available: {len(loader)}")
    
    # Limit number of frames
    num_frames = min(num_frames, len(loader))
    print(f"Processing first {num_frames} frames...")
    
    # Initialize pipeline
    print("\nInitializing object classification pipeline...")
    pipeline = ObjectClassificationPipeline('yolov8n.pt')
    
    print("\n" + "="*60)
    print("STARTING DEMO")
    print("="*60)
    
    for i in range(num_frames):
        # Load frame pair
        rgb, depth = loader.get_frame_pair(i)
        
        if rgb is None or depth is None:
            print(f"Skipping frame {i} - failed to load")
            continue
        
        # Process through pipeline
        results = pipeline.process_frame(rgb, depth)
        
        # Print results every 10 frames
        if i % 10 == 0 or len(results['prioritized_obstacles']) > 0:
            print(f"\n{'='*60}")
            print(f"FRAME {i+1}/{num_frames}")
            print(f"{'='*60}")
            print(f"Raw detections: {len(results['raw_detections'])}")
            print(f"Tracked objects: {len(results['tracked_objects'])}")
            
            # Print top obstacles
            if results['prioritized_obstacles']:
                print("\n🎯 TOP THREATS:")
                for idx, obs in enumerate(results['prioritized_obstacles'][:3]):
                    print(f"  {idx+1}. {obs['class'].upper()} "
                          f"[Track ID: {obs.get('track_id', 'N/A')}]")
                    print(f"     Zone: {obs['zone'].upper()} | "
                          f"Depth: {obs['depth']:.2f}m | "
                          f"Threat: {obs['threat_score']:.1f}")
                    print(f"     Action: {obs['suggested_action']}")
                    if obs.get('is_stable'):
                        print(f"     ✓ Stable track (age: {obs.get('age', 0)} frames)")
            
            # Print audio cues
            print("\n🔊 AUDIO CUES:")
            for zone, cue in results['audio_cues'].items():
                if cue['active']:
                    print(f"  {zone.upper()}: "
                          f"{cue['frequency']:.0f}Hz | "
                          f"Vol: {cue['volume']:.2f} | "
                          f"Type: {cue['object_type']} | "
                          f"Urgency: {cue['urgency']}")
                else:
                    print(f"  {zone.upper()}: 🔇 Clear")
            
            # Critical alert
            if results['critical_alert']:
                print(f"\n🚨 {results['critical_alert']}")
        
        # Save visualization
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(output_path, results['visualization_frame'])
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"\n✓ Processed {i+1}/{num_frames} frames...")
    
    print(f"\n{'='*60}")
    print("✅ DEMO COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}/")
    print(f"Total frames processed: {num_frames}")
    print(f"\nNext steps:")
    print(f"  1. Check {output_dir}/ for annotated images")
    print(f"  2. Use these for your presentation slides")
    print(f"  3. Optional: Create video with:")
    print(f"     ffmpeg -framerate 5 -i {output_dir}/frame_%04d.png demo_video.mp4")

if __name__ == "__main__":
    # Path to your downloaded TUM dataset
    dataset_path = "rgbd_dataset_freiburg1_desk"
    
    # Run demo on first 50 frames
    run_tum_demo(
        dataset_path=dataset_path,
        num_frames=50,
        output_dir='demo_output'
    )
