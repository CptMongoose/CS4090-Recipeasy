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

class TestLoadRecipeData:
    
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_load_recipe_data_first_path(self, mock_read_csv, mock_exists, sample_dataframe):
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_dataframe
        
        result = load_recipe_data()
        
        assert result is not None
        assert len(result) == 3
        mock_read_csv.assert_called_once()
    
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_load_recipe_data_second_path(self, mock_read_csv, mock_exists, sample_dataframe):
        mock_exists.side_effect = [False, True]
        mock_read_csv.return_value = sample_dataframe
        
        result = load_recipe_data()
        
        assert result is not None
        assert len(result) == 3
    
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_load_recipe_data_csv_error_then_success(self, mock_read_csv, mock_exists, sample_dataframe):
        mock_exists.return_value = True
        mock_read_csv.side_effect = [Exception("Read error"), sample_dataframe]
        
        result = load_recipe_data()
        
        assert result is not None
        assert len(result) == 3
    
    @patch('os.path.exists')
    @patch('builtins.input')
    @patch('Recipeasy.download_dataset')
    def test_load_recipe_data_download_yes_success(self, mock_download, mock_input, mock_exists, sample_dataframe):
        mock_exists.side_effect = [False, False, False, False, True]
        mock_input.return_value = 'yes'
        mock_download.return_value = True
        
        with patch('pandas.read_csv', return_value=sample_dataframe):
            result = load_recipe_data()
            
            assert result is not None
            mock_download.assert_called_once()
    
    @patch('os.path.exists')
    @patch('builtins.input')
    @patch('Recipeasy.download_dataset')
    def test_load_recipe_data_download_yes_failure(self, mock_download, mock_input, mock_exists):
        mock_exists.return_value = False
        mock_input.return_value = 'yes'
        mock_download.return_value = False
        
        result = load_recipe_data()
        
        assert result is None
        mock_download.assert_called_once()
    
    @patch('os.path.exists')
    @patch('builtins.input')
    def test_load_recipe_data_download_no(self, mock_input, mock_exists):
        mock_exists.return_value = False
        mock_input.return_value = 'no'
        
        result = load_recipe_data()
        
        assert result is None

class TestSearchRecipesByName:
    
    def test_search_recipes_by_name_found(self, sample_dataframe):
        result = search_recipes_by_name(sample_dataframe, 'chocolate')
        assert len(result) == 1
        assert 'Chocolate Cake' in result['name'].values
    
    def test_search_recipes_by_name_case_insensitive(self, sample_dataframe):
        result = search_recipes_by_name(sample_dataframe, 'CHOCOLATE')
        assert len(result) == 1
    
    def test_search_recipes_by_name_not_found(self, sample_dataframe):
        result = search_recipes_by_name(sample_dataframe, 'pizza')
        assert len(result) == 0
    
    def test_search_recipes_by_name_partial_match(self, sample_dataframe):
        result = search_recipes_by_name(sample_dataframe, 'cake')
        assert len(result) == 1


class TestSearchRecipesByIngredient:
    
    def test_search_recipes_by_ingredient_found(self, sample_dataframe):
        result = search_recipes_by_ingredient(sample_dataframe, 'chicken')
        assert len(result) == 1
        assert 'Chicken Soup' in result['name'].values
    
    def test_search_recipes_by_ingredient_multiple(self, sample_dataframe):
        result = search_recipes_by_ingredient(sample_dataframe, 'flour, sugar')
        assert len(result) == 2
    
    def test_search_recipes_by_ingredient_not_found(self, sample_dataframe):
        result = search_recipes_by_ingredient(sample_dataframe, 'bacon')
        assert len(result) == 0
    
    def test_search_recipes_by_ingredient_no_column(self):
        df = pd.DataFrame({'name': ['Test']})
        result = search_recipes_by_ingredient(df, 'test')
        assert len(result) == 0
    
    def test_search_recipes_by_ingredient_empty_query(self, sample_dataframe):
        result = search_recipes_by_ingredient(sample_dataframe, '')
        assert len(result) == 0
    
    def test_search_recipes_by_ingredient_whitespace_only(self, sample_dataframe):
        result = search_recipes_by_ingredient(sample_dataframe, '   ,  ,  ')
        assert len(result) == 0

