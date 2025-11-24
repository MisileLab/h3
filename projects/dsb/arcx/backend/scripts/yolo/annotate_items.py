"""Simple annotation tool for item detection.

This script helps create annotations for extraction screenshots.
For more advanced annotation, consider using tools like LabelImg or CVAT.
"""

import argparse
import json
from pathlib import Path

import cv2

from arcx.valuation.config import ITEM_TYPES, RARITY_LEVELS

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAnnotator:
    """Simple bounding box annotator."""

    def __init__(self, images_dir: Path, output_file: Path):
        self.images_dir = images_dir
        self.output_file = output_file
        self.annotations = {}

        # Load existing annotations if available
        if output_file.exists():
            with open(output_file) as f:
                self.annotations = json.load(f)
            logger.info(f"Loaded existing annotations: {len(self.annotations)} images")

        self.current_image = None
        self.current_filename = None
        self.bboxes = []
        self.drawing = False
        self.start_point = None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.current_image.copy()
                cv2.rectangle(img_copy, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Annotator", img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)

            # Add bbox
            x1 = min(self.start_point[0], end_point[0])
            y1 = min(self.start_point[1], end_point[1])
            x2 = max(self.start_point[0], end_point[0])
            y2 = max(self.start_point[1], end_point[1])

            if x2 - x1 > 10 and y2 - y1 > 10:  # Minimum size
                self.bboxes.append([x1, y1, x2, y2])
                logger.info(f"Added bbox: [{x1}, {y1}, {x2}, {y2}]")

            # Redraw
            self.draw_bboxes()

    def draw_bboxes(self):
        """Draw current bounding boxes."""
        img_copy = self.current_image.copy()
        for i, bbox in enumerate(self.bboxes):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_copy,
                f"#{i}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        cv2.imshow("Annotator", img_copy)

    def annotate_image(self, image_path: Path):
        """Annotate a single image."""
        self.current_filename = image_path.name
        self.current_image = cv2.imread(str(image_path))

        if self.current_image is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # Load existing annotations if any
        if self.current_filename in self.annotations:
            logger.info(f"Image already annotated: {self.current_filename}")
            response = input("Re-annotate? (y/n): ")
            if response.lower() != "y":
                return

        self.bboxes = []

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", self.mouse_callback)

        logger.info(f"Annotating: {self.current_filename}")
        logger.info("Draw bounding boxes with mouse. Press:")
        logger.info("  'u' - Undo last box")
        logger.info("  's' - Save and next image")
        logger.info("  'q' - Quit without saving")
        logger.info("  'n' - Skip to next image")

        self.draw_bboxes()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("u"):  # Undo
                if self.bboxes:
                    self.bboxes.pop()
                    logger.info("Removed last bbox")
                    self.draw_bboxes()

            elif key == ord("s"):  # Save
                if not self.bboxes:
                    logger.warning("No bboxes to save")
                    continue

                # Prompt for item types and rarities
                items = []
                for i, bbox in enumerate(self.bboxes):
                    print(f"\nBox #{i}: {bbox}")
                    print(f"Item types: {', '.join(ITEM_TYPES)}")
                    item_type = input("Item type: ").strip()

                    print(f"Rarities: {', '.join(RARITY_LEVELS)}")
                    rarity = input("Rarity: ").strip()

                    if item_type in ITEM_TYPES and rarity in RARITY_LEVELS:
                        items.append({
                            "item_type": item_type,
                            "rarity": rarity,
                            "bbox": bbox,
                        })
                    else:
                        logger.warning(f"Invalid item type or rarity, skipping box #{i}")

                if items:
                    h, w = self.current_image.shape[:2]
                    self.annotations[self.current_filename] = {
                        "items": items,
                        "image_width": w,
                        "image_height": h,
                    }
                    self.save_annotations()
                    logger.info(f"Saved {len(items)} items")

                cv2.destroyWindow("Annotator")
                return True

            elif key == ord("n"):  # Skip
                cv2.destroyWindow("Annotator")
                return True

            elif key == ord("q"):  # Quit
                cv2.destroyWindow("Annotator")
                return False

        return True

    def save_annotations(self):
        """Save annotations to file."""
        with open(self.output_file, "w") as f:
            json.dump(self.annotations, f, indent=2)
        logger.info(f"Saved annotations to: {self.output_file}")

    def run(self):
        """Run annotation tool."""
        image_files = sorted(self.images_dir.glob("*.png")) + \
                      sorted(self.images_dir.glob("*.jpg"))

        if not image_files:
            logger.error(f"No images found in {self.images_dir}")
            return

        logger.info(f"Found {len(image_files)} images")

        for image_path in image_files:
            if not self.annotate_image(image_path):
                break

        logger.info("Annotation complete!")
        logger.info(f"Total annotated: {len(self.annotations)} images")


def main():
    parser = argparse.ArgumentParser(description="Annotate extraction screenshots")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing extraction screenshots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations.json"),
        help="Output JSON file for annotations",
    )

    args = parser.parse_args()

    if not args.images.exists():
        logger.error(f"Images directory not found: {args.images}")
        return

    annotator = SimpleAnnotator(args.images, args.output)
    annotator.run()


if __name__ == "__main__":
    main()
