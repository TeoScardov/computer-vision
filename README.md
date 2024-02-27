# README: Binary Morphological Operations and Sudoku Recognition
## Dependencies
- Python 3.x
- OpenCV (cv2) library
- NumPy library
- Matplotlib library

Ensure you have the above prerequisites installed in your environment. You can install them using pip:
```bash
pip install numpy opencv-python matplotlib
```
## Notebooks
The two programs are available both in .py format for CLI and in notebook format for easier use and visualization.
## Morphological Operations
### Overview
This program is a command-line interface (CLI) tool designed for performing morphological operations on images, including erosion, dilation, opening, and closing. 
### Usage
To use the program, you must provide three arguments to the command line: the path to the image file, the path to the structuring element image file, and the operation to be performed.
```bash
python MorphOps.py [image_path] [structuring_element_path] [operation]
```
- image_path: Required. The path to the image file on which the morphological operation will be performed.
- structuring_element_path: Required. The path to the binary image file that represents the structuring element.
- operation: Required. Specifies the morphological operation to perform. Use 'e' for erosion, 'd' for dilation, 'c' for closing, and 'o' for opening.

Example:
```bash
python MorphOps.py image.png 3x3.png e
```
### Functionality
- Erosion (e): This operation erodes away the boundaries of the foreground object.
- Dilation (d): Dilation adds pixels to the boundaries of objects in an image.
- Closing (c): A closing operation is a dilation followed by an erosion, useful for closing small holes inside the foreground objects.
- Opening (o): An opening operation is an erosion followed by a dilation, useful for removing small objects from the foreground.

When the specified operation is performed, the program displays two images side by side: the original image and the result of the morphological operation.

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
### Functionality
- Preprocessing: The image is converted to grayscale and then to a binary image using adaptive thresholding to highlight the grid and numbers.
- Line Detection: Uses Hough line detection to find the grid lines and isolate the sudoku cells.
- Number Detection: Recognizes numbers in each cell by comparing with templates.
- Output: Prints the solved Sudoku grid to the console.

Press Enter to exit the program after viewing the output.
