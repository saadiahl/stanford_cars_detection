import os
import pandas as pd
import cv2
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class YOLOConverter:
    def __init__(self, dataset_root: str, unify_classes: bool = False):
        self.dataset_root = dataset_root
        self.unify_classes = unify_classes  # New parameter to unify classes
        self.train_annos = os.path.join(dataset_root, "train_annotations.csv")
        self.test_annos = os.path.join(dataset_root, "cars_test_annos_withlabels.csv")
        self.class_meta = os.path.join(dataset_root, "cars_meta.csv")
        self.train_images = os.path.join(dataset_root, "cars_train")
        self.test_images = os.path.join(dataset_root, "cars_test")

        # Define YOLO directory structure
        self.yolo_train_images = os.path.join(dataset_root, "images/yolo_train")
        self.yolo_val_images = os.path.join(dataset_root, "images/yolo_val")
        self.yolo_test_images = os.path.join(dataset_root, "images/yolo_test")

        self.yolo_train_labels = os.path.join(dataset_root, "labels/yolo_train")
        self.yolo_val_labels = os.path.join(dataset_root, "labels/yolo_val")
        self.yolo_test_labels = os.path.join(dataset_root, "labels/yolo_test")

        self.yolo_config = os.path.join(dataset_root, "data.yaml")

        self.class_mapping = self.load_class_mapping()

        # Ensure required directories exist
        for path in [
            self.yolo_train_images,
            self.yolo_val_images,
            self.yolo_test_images,
            self.yolo_train_labels,
            self.yolo_val_labels,
            self.yolo_test_labels,
        ]:
            os.makedirs(path, exist_ok=True)

    def load_class_mapping(self):
        """Loads class_id to class_name mapping from metadata."""
        if os.path.exists(self.class_meta):
            return (
                pd.read_csv(self.class_meta)
                .set_index("class_id")["class_name"]
                .to_dict()
            )
        return {}

    @staticmethod
    def convert_bbox(x1, y1, x2, y2, img_width, img_height):
        """Converts bounding box from absolute to YOLO format."""
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return x_center, y_center, width, height

    def process_annotations(
        self, df, image_dir: str, label_output_dir: str, image_output_dir: str
    ):
        """Converts annotations to YOLO format and moves images to correct directories."""
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = os.path.join(image_dir, row["filename"])
            if not os.path.exists(img_path):
                print(f"Warning: Image {row['filename']} not found in {image_dir}.")
                continue

            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]
            x_center, y_center, width, height = self.convert_bbox(
                row["x1"], row["y1"], row["x2"], row["y2"], img_width, img_height
            )

            class_id = (
                0 if self.unify_classes else row.get("class_id", 0) - 1
            )  # Unify all classes if enabled
            label_filename = os.path.join(
                label_output_dir, row["filename"].replace(".jpg", ".txt")
            )
            with open(label_filename, "w") as f:
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )

            # Move the image to the correct directory
            shutil.move(img_path, os.path.join(image_output_dir, row["filename"]))

    def generate_yaml(self):
        """Creates the YOLO dataset configuration file."""
        class_names = (
            ["car"] if self.unify_classes else list(self.class_mapping.values())
        )
        yaml_content = f"""
train: {self.yolo_train_images}
val: {self.yolo_val_images}
test: {self.yolo_test_images}
names: {class_names}
        """
        with open(self.yolo_config, "w") as f:
            f.write(yaml_content)

    def split_and_convert(self):
        """Splits test data into validation and test, then converts annotations."""
        test_df = pd.read_csv(self.test_annos)
        test_df["class_id"] -= 1  # Adjust class_id to start from 0

        val_df, test_df = train_test_split(
            test_df, test_size=0.5, stratify=test_df["class_id"], random_state=42
        )

        self.process_annotations(
            val_df, self.test_images, self.yolo_val_labels, self.yolo_val_images
        )
        self.process_annotations(
            test_df, self.test_images, self.yolo_test_labels, self.yolo_test_images
        )
        self.process_annotations(
            pd.read_csv(self.train_annos),
            self.train_images,
            self.yolo_train_labels,
            self.yolo_train_images,
        )

        self.generate_yaml()

    def convert(self):
        """Executes the full conversion process."""
        self.split_and_convert()


if __name__ == "__main__":
    dataset_root = "/stanford_cars"
    converter = YOLOConverter(
        dataset_root, unify_classes=True
    )  # Set to True to unify all classes into 'car'
    converter.convert()
