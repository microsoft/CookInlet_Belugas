from PytorchWildlife.data.bioacoustics.bioacoustics_annotations import BaseReader, AnnotationCreator
from datetime import datetime
import pandas as pd
import argparse
import csv
import os

class WhaleSpeciesReader(BaseReader):
    def __init__(self, data_path, species):
        super().__init__(data_path)
        self.data_path = data_path
        self.species = species
        self.annotation_csv_path = os.path.join(
            data_path, f"{species}_annotations_processed.csv"
        )
        self.output_path = os.path.join(data_path, f"{species}_annotations.json")

    def add_dataset_info(self):
        self.annotation_creator.add_info(
            title=f"NOAA_{self.species}",
            description=f"{self.species} call dataset with annotations",
            version="1.0",
            publication_date=datetime.now().strftime("%Y%m%d"),
        )

    def add_categories(self):
        df = pd.DataFrame([{"name": self.species}])
        self.annotation_creator.add_categories(df)

    def add_sounds(self):
        print(f"Reading annotations from {self.annotation_csv_path}")
        if not os.path.exists(self.annotation_csv_path):
            print("Annotation CSV not found!")
            return  # Stop processing if annotations are not found

        next_id = 0
        self.audio_id_map = {}
        with open(self.annotation_csv_path, "r", newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["audiofile_path"]
                if rel_path not in self.audio_id_map:
                    self.audio_id_map[rel_path] = next_id
                    file_path = os.path.join(self.data_path, rel_path)
                    duration, sample_rate = self.annotation_creator._get_duration_and_sample_rate(file_path)
                    self.annotation_creator.add_sound(
                        id=next_id,
                        file_name_path=file_path,
                        duration=duration,
                        sample_rate=sample_rate,
                        latitude=None,
                        longitude=None,
                        location=row['location']
                    )
                    next_id += 1

    def add_annotations(self):
        with open(self.annotation_csv_path, "r", newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for anno_id, row in enumerate(reader):
                path = row["audiofile_path"]
                sound_id = self.audio_id_map[path]
                t_min = float(row["startSeconds"])
                t_max = t_min + float(row["durationSeconds"])
                f_min = float(row["lowFreq"])
                f_max = float(row["highFreq"])
                category_match = [cat for cat in self.annotation_creator.data["categories"] if cat["name"] == row["Species"] or cat["name"][0] == row["Species"]]
                if not category_match:
                    continue
                category_id = category_match[0]["id"]
                category = category_match[0]["name"]
                self.annotation_creator.add_annotation(
                    anno_id=anno_id,
                    sound_id=sound_id,
                    category_id=category_id,
                    category=category,
                    t_min=t_min,
                    t_max=t_max,
                    f_min=f_min,
                    f_max=f_max,
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process whale species annotations")
    parser.add_argument(
        "--species",
        type=str,
        required=True,
        choices=["Beluga", "Humpback", "Orca"],
        help="Species to process (Beluga, Humpback, or Orca)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="NOAA_Whales",
        help="Path to the dataset directory (default: NOAA_Whales)"
    )
    
    args = parser.parse_args()
    
    reader = WhaleSpeciesReader(args.data_path, args.species)
    reader.process_dataset()