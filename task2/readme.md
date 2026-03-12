# Task 2 – Perception System for RoboGambit

---

## Introduction

In this task, we developed the **perception module** for the RoboGambit system.  
The goal of this module is to analyze an image of the physical chess board and automatically determine the **current board configuration**.

To achieve this, we used **ArUco markers** placed on the chess pieces and on the four corners of the board. By detecting these markers in the image, the system can identify where each piece is located and map it to the correct square of the **6 × 6 chess board**.

This allows the robot system to observe the board using a camera, understand the game state, and pass that information to the game engine for decision making.

---

## Libraries Used

The perception system was implemented in **Python** using a few commonly used libraries for computer vision and numerical processing.

- **OpenCV** – used for image processing and ArUco marker detection  
- **NumPy** – used for handling matrices and representing the board state  
- **sys** – used for handling command line inputs when running the program

These libraries make it easier to process images, perform coordinate transformations, and manage board data efficiently.

---

## Camera Calibration

Before analyzing the board image, the camera parameters must be defined so that the system can correctly interpret the captured image.

These parameters describe properties of the camera such as its focal length and optical center. Using these values, the captured image can be corrected for distortions.

The system first performs **image undistortion**, which ensures that straight lines on the board appear straight in the processed image. This step improves the accuracy of marker detection and coordinate calculations.

---

## ArUco Marker Detection

The system uses **ArUco markers** to detect important elements in the scene.

These markers are small square patterns that can be easily recognized by computer vision algorithms. Each marker has a unique ID, which allows the system to distinguish between different pieces and board corners.

When the program processes the image, it scans the grayscale version of the image to detect all visible markers. The detected markers can also be highlighted visually on the image so that it is easy to verify whether detection is working correctly.

For example, after detection, the program can display the image with bounding boxes drawn around each marker.

---

## Board Corner Detection

To understand the orientation and position of the chess board, four special markers are placed at the corners of the board.

These markers define the coordinate system of the board.

| Marker ID | Board Corner |
|-----------|-------------|
| 21 | Top Left |
| 22 | Bottom Left |
| 23 | Bottom Right |
| 24 | Top Right |

By detecting these four markers, the system can determine the boundaries of the board and calculate how the board appears in the camera image.

---

## Pixel to World Coordinate Transformation

Once the board corners are detected, the next step is to convert **pixel coordinates from the image into board coordinates**.

This is done by computing a transformation between the image plane and the board plane. Using this transformation, the position of each detected marker can be translated into real board coordinates.

For example, a marker detected near the center of the image may correspond to a square near the middle of the chess board.

This transformation is an important step because it allows the system to accurately determine which square each piece belongs to.

---

## Board Representation

After detecting all pieces, the system constructs a **6 × 6 board representation**.

The board is stored as a grid where each cell represents one square of the chess board. Each cell contains a number representing the piece located on that square.

For example, a detected board configuration might look like:
[[0 0 0 0 0 0]
[0 1 0 0 0 0]
[0 0 0 7 0 0]
[0 0 0 0 0 0]
[0 0 3 0 0 0]
[0 0 0 0 0 0]]


Here:

- `0` represents an empty square  
- other numbers correspond to specific chess pieces  

This board structure can then be directly passed to the game engine.

---

## Mapping Pieces to Board Squares

After calculating the world coordinates of each marker, the system determines which square of the board it belongs to.

This is done by dividing the board area into equal-sized squares and calculating the row and column for each detected marker.

For example, if a marker is detected near the center of the board, it may correspond to a square such as **C3 or D3** depending on its exact position.

To ensure stability, the calculated row and column values are constrained so that they always remain within the **6 × 6 board limits**.

Once the correct square is determined, the corresponding piece ID is stored in the board representation.

---

## Visualizing the Board

To make it easier to verify the results, the system also generates a **visual board representation**.

A grid is drawn to represent the chess board, and the detected piece IDs are displayed inside their corresponding squares.

This visual output helps confirm that the pieces are mapped to the correct locations.

For example, the visualization may show a grid where each square contains a number representing the detected piece.

---

## Running the Program

The perception module is designed to run from the command line by providing an image of the chess board.

When the program runs, it performs the following steps:

1. Loads the input image  
2. Detects all ArUco markers  
3. Identifies the board corners  
4. Converts marker positions to board coordinates  
5. Reconstructs the board configuration  

This process allows the system to automatically interpret the board state from a single image.

---

## Output

After processing the image, the system displays two visual outputs:

**Detected Markers**

This window shows the original camera image with all detected markers highlighted.

**Game Board**

This window displays the reconstructed board state as a grid with piece identifiers.

These visualizations help verify whether the detection and mapping steps are working correctly.

---

## Conclusion

In this task, we developed a perception module capable of detecting a physical chess board configuration using **ArUco markers**.

The system processes camera images, identifies marker locations, converts those positions into board coordinates, and reconstructs the full board state.

This module acts as the bridge between the **physical chess board and the RoboGambit game engine**, allowing the robot to understand the real-world board configuration and make intelligent decisions based on the detected positions.

