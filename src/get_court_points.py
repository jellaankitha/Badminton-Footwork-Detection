import cv2

image_path = "output/court_samples/frame_4800s.jpg"

image = cv2.imread(image_path)

if image is None:
    raise RuntimeError("Could not open frame_4800s.jpg")

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:
            points.append((x, y))

            print(f"Point {len(points)}: ({x}, {y})")

            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

            cv2.putText(
                image,
                str(len(points)),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            cv2.imshow("Select Court Corners", image)

            if len(points) == 4:
                print()
                print("COURT POINTS:")
                print(points)
                print()
                print("Order used:")
                print("1 = top-left")
                print("2 = top-right")
                print("3 = bottom-right")
                print("4 = bottom-left")
                print()
                print("Press ESC to close.")

cv2.namedWindow("Select Court Corners")
cv2.setMouseCallback("Select Court Corners", click_event)

print("Click the 4 corners of the COURT.")
print()
print("Click in this order:")
print("1. TOP-LEFT")
print("2. TOP-RIGHT")
print("3. BOTTOM-RIGHT")
print("4. BOTTOM-LEFT")
print()
print("Do NOT click the outer red floor.")
print("Click the four corners where the badminton court boundary is visible.")

cv2.imshow("Select Court Corners", image)

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

cv2.destroyAllWindows()