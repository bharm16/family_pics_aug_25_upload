# family_pics_aug_25_upload

Utility script for correcting the orientation of large batches of photos.

### Features

- Skips network calls when EXIF orientation metadata is present.
- Uses a small downscaled copy of each image for the Vision API to reduce upload time.
- Lightweight local face-detection heuristic to avoid unnecessary API calls.
- Concurrent processing with a higher default worker count.
