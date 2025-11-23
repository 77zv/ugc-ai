import os
import glob
import shutil
from dotenv import load_dotenv

# Import script processing functions
from script_repurposer import build_personality_db, repurpose_script, DB_PATH
from video_processor import process_video, scenes_to_script

# === Load API Key ===
load_dotenv()

# === MAIN ===

if __name__ == "__main__":
    # Ensure personality database exists
    if not os.path.exists(DB_PATH):
        build_personality_db()
    else:
        print(f"Personality DB found at {DB_PATH}/")

    # Display menu
    print("\n" + "="*60)
    print("UGC AI SCRIPT PERSONALIZER")
    print("="*60)
    print("\nOptions:")
    print("1. Process video (extract & personalize)")
    print("2. Process raw script from raw_scripts/ folder")
    print("3. Personalize existing script.txt")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # === Option 1: Process Video ===
        videos_folder = "videos"
        
        # Check for videos folder and list available videos
        if os.path.exists(videos_folder):
            video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.webm']
            video_files = []
            for ext in video_extensions:
                pattern = os.path.join(videos_folder, ext)
                video_files.extend(glob.glob(pattern))
            
            if video_files:
                print(f"\n📹 Found {len(video_files)} video(s) in videos/ folder:")
                print("-" * 60)
                for i, video in enumerate(video_files, 1):
                    filename = os.path.basename(video)
                    print(f"{i}. {filename}")
                
                print("-" * 60)
                selection = input(f"\nSelect video (1-{len(video_files)}) or enter custom path: ").strip()
                
                # Check if it's a number selection
                if selection.isdigit() and 1 <= int(selection) <= len(video_files):
                    video_path = video_files[int(selection) - 1]
                else:
                    # Custom path
                    video_path = selection
            else:
                print(f"\n⚠️  No videos found in {videos_folder}/ folder")
                video_path = input("Enter path to video file: ").strip()
        else:
            print(f"\n💡 Tip: Create a 'videos/' folder to see video list")
            video_path = input("Enter path to video file: ").strip()
        
        # Process the video if it exists
        if os.path.exists(video_path):
            # Create raw_scripts folder if it doesn't exist
            raw_scripts_dir = "raw_scripts"
            os.makedirs(raw_scripts_dir, exist_ok=True)
            
            # Process video
            scenes = process_video(video_path)
            
            # Generate a filename based on the video name
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            raw_script_path = os.path.join(raw_scripts_dir, f"{video_name}_raw.txt")
            
            # Save raw extracted script to raw_scripts folder
            scenes_to_script(scenes, raw_script_path)
            print(f"\n💾 Raw script saved to: {raw_script_path}")
            
            # Also save to script.txt for processing
            scenes_to_script(scenes, "script.txt")
            
            print("\n" + "="*60)
            print("Video processed! Now personalizing script...")
            print("="*60)
            
            # Personalize the generated script
            repurpose_script()
        else:
            print(f"Error: Video file not found: {video_path}")
    
    elif choice == "2":
        # === Option 2: Process Raw Script ===
        raw_scripts_dir = "raw_scripts"
        
        if os.path.exists(raw_scripts_dir):
            raw_scripts = glob.glob(os.path.join(raw_scripts_dir, "*.txt"))
            
            if raw_scripts:
                print(f"\n📝 Found {len(raw_scripts)} raw script(s):")
                print("-" * 60)
                for i, script_path in enumerate(raw_scripts, 1):
                    filename = os.path.basename(script_path)
                    print(f"{i}. {filename}")
                
                print("-" * 60)
                selection = input(f"\nSelect raw script (1-{len(raw_scripts)}): ").strip()
                
                if selection.isdigit() and 1 <= int(selection) <= len(raw_scripts):
                    selected_script = raw_scripts[int(selection) - 1]
                    
                    # Copy selected raw script to script.txt for processing
                    shutil.copy(selected_script, "script.txt")
                    print(f"\n✅ Loaded: {os.path.basename(selected_script)}")
                    
                    print("\n" + "="*60)
                    print("Now personalizing script...")
                    print("="*60)
                    
                    # Personalize the script
                    repurpose_script()
                else:
                    print("Invalid selection")
            else:
                print(f"\n⚠️  No raw scripts found in {raw_scripts_dir}/ folder")
                print("💡 Tip: Process a video first (option 1) to create raw scripts")
        else:
            print(f"\n⚠️  {raw_scripts_dir}/ folder not found")
            print("💡 Tip: Process a video first (option 1) to create the folder")
    
    else:
        # === Option 3: Direct Script Personalization ===
        repurpose_script()

    # Optional: quick regenerate loop
    while input("\nRegenerate? (y/n): ").strip().lower() == "y":
        repurpose_script()
