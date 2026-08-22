for f in *.mp4; do
  # Crop and set to 30 fps
  # Also restrict duration to first 25 seconds (if shorter, no effect)
  ffmpeg -i "$f" -vf "crop=720:720:70:in_h-720-70" -r 30 -t 25 "cropped_$f"
done