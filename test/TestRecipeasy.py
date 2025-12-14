import sys
import pytest
import pandas as pd
import random
import os
import json
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.Recipeasy import download_dataset
from src.Recipeasy import load_recipe_data
from src.Recipeasy import search_recipes_by_name
from src.Recipeasy import search_recipes_by_ingredient
from src.Recipeasy import display_recipe
from src.Recipeasy import main

