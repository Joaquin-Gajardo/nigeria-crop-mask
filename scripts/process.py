import sys
from pathlib import Path

sys.path.append("..")

from src.processors import (
    GeoWikiProcessor,
    TogoProcessor,
    NigeriaProcessor,
)


def process_geowiki():
    processor = GeoWikiProcessor(Path("../data"))
    processor.process()


def process_togo():
    processor = TogoProcessor(Path("../data"))
    processor.process()

def process_nigeria():
    processor = NigeriaProcessor(Path("../data"))
    processor.process()


if __name__ == "__main__":
    #process_geowiki()
    #process_togo()
    process_nigeria()
