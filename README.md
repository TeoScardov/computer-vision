# README: Mathematical Operations and Sudoku Recognition
## Dependencies
- Python 3.x
- OpenCV (cv2) library
- NumPy library
- Matplotlib library

Ensure you have the above prerequisites installed in your environment. You can install them using pip:
```bash
pip install numpy opencv-python matplotlib
```
## Sudoku Recognition
### Overview
This program is a command-line interface (CLI) tool designed to detect Sudoku puzzles from images.
### Usage
To use this program, you must provide at least one argument to the command line: the path to the image file containing the Sudoku puzzle. Optionally, you can also specify a custom path for templates used in number recognition.

```bash
python Sudoku.py [image_path] [template_path]
```
- image_path: Required. The path to the image file containing the Sudoku puzzle you wish to recognize.
- template_path: Optional. The path to the directory containing templates for number recognition. If not specified, a default path of "templates/" will be used. The templates must be png files of the digits labeled 1.png through 9.png.

Example:
```bash
python Sudoku.py puzzle.png
```
Or with a custom template path:
```bash
python Sudoku.py puzzle.png custom_templates/
```
## Functionality
- Preprocessing: The image is converted to grayscale and then to a binary image using adaptive thresholding to highlight the grid and numbers.
- Line Detection: Uses Hough line detection to find the grid lines and isolate the sudoku cells.
- Number Detection: Recognizes numbers in each cell by comparing with templates.
- Output: Prints the solved Sudoku grid to the console.

Press Enter to exit the program after viewing the output.
