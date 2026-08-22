# Loop over all directories
for dir in */; do
# Run crop and rename
    (cd "$dir" && bash ../crop_videos.sh && bash ../rename_cropped_videos.sh )
    # cd ..
done