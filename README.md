# Oral Lesion Screening Demo (OpenCV + Streamlit)

Classical computer-vision pipeline, no training data required:
1. Resize, bilateral denoise, CLAHE contrast enhancement (LAB L-channel)
2. Red patches: LAB a-channel redness above the image median; white patches: HSV low-saturation, bright (leukoplakia-like)
3. Morphological open/close, contour extraction
4. Per-region area, circularity, solidity, border irregularity
5. Heuristic "attention level" (Low / Moderate / High)

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
## Live url
```
https://oralcancerdetector.streamlit.app/
```
## Files
- `processing.py`: OpenCV pipeline (importable, testable)
- `app.py`: Streamlit interface with live parameter sliders

## Limitations
Colour heuristics cannot tell cancer from infection, trauma, teeth, or glare. This is a
teaching prototype, not a diagnostic tool. To make it clinically meaningful, replace the
heuristic score with a CNN trained on labelled data (see the project plan) and keep this
pipeline for preprocessing and visualisation.

## Ideas to extend
- Add a ROI crop tool (streamlit-cropper) to exclude teeth
- Plug in a PyTorch model and Grad-CAM overlay
- Batch mode for folders; export CSV reports
-e-websolutions tools owner
