import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import io

path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(path))

from src.Recipeasy import (
    download_dataset,
    load_recipe_data,
    search_recipes_by_name,
    search_recipes_by_ingredient,
    display_recipe,
    main
)

@pytest.fixture
def sample_dataframe():
    data = {
        'name': ['Chocolate Cake', 'Vanilla Cookies', 'Chicken Soup'],
        'minutes': [45, 30, 60],
        'n_steps': [10, 8, 12],
        'n_ingredients': [8, 6, 10],
        'ingredients': [
            "['flour', 'sugar', 'chocolate', 'eggs']",
            "['flour', 'sugar', 'vanilla', 'butter']",
            "['chicken', 'carrots', 'celery', 'onion']"
        ],
        'steps': [
            "['Mix ingredients', 'Bake at 350F']",
            "['Mix', 'Shape', 'Bake']",
            "['Boil water', 'Add ingredients', 'Simmer']"
        ],
        'description': [
            'Delicious chocolate cake',
            'Sweet vanilla cookies',
            'Hearty chicken soup'
        ]
    }
    return pd.DataFrame(data)

class TestDownloadDataset:
    
    @patch('Recipeasy.kaggle')
    def test_download_dataset_success(self, mock_kaggle):
        mock_kaggle.api.dataset_download_files.return_value = None
        result = download_dataset()
        assert result is True
        mock_kaggle.api.dataset_download_files.assert_called_once()
    
    def test_download_dataset_import_error(self):
        with patch.dict('sys.modules', {'kaggle': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                result = download_dataset()
                assert result is False
    
    @patch('Recipeasy.kaggle')
    def test_download_dataset_exception(self, mock_kaggle):
        mock_kaggle.api.dataset_download_files.side_effect = Exception("Download failed")
        result = download_dataset()
        assert result is False