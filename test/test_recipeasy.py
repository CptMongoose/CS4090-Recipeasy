import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import io

path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(path))

import Recipeasy

from Recipeasy import (
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

class TestDisplayRecipe:
    
    def test_display_recipe_all_fields(self, sample_dataframe, capsys):
        recipe = sample_dataframe.iloc[0]
        display_recipe(recipe)
        captured = capsys.readouterr()
        
        assert 'Chocolate Cake' in captured.out
        assert '45 minutes' in captured.out
        assert '10' in captured.out
        assert '8' in captured.out
    
    def test_display_recipe_missing_fields(self, capsys):
        recipe = pd.Series({'name': 'Simple Recipe'})
        display_recipe(recipe)
        captured = capsys.readouterr()
        
        assert 'Simple Recipe' in captured.out
        assert 'RECIPE RECOMMENDATION' in captured.out
    
    def test_display_recipe_na_values(self, capsys):
        recipe = pd.Series({
            'name': 'Test Recipe',
            'ingredients': pd.NA,
            'steps': None,
            'description': pd.NA
        })
        display_recipe(recipe)
        captured = capsys.readouterr()
        
        assert 'Test Recipe' in captured.out

class TestMain:
    
    @patch('Recipeasy.load_recipe_data')
    def test_main_no_data(self, mock_load, capsys):
        mock_load.return_value = None
        main()
        captured = capsys.readouterr()
        assert 'RANDOM RECIPE RECOMMENDER' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_surprise_me(self, mock_input, mock_load, sample_dataframe, capsys):
        mock_load.return_value = sample_dataframe
        mock_input.side_effect = ['1', '4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'RECIPE RECOMMENDATION' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_search_by_name_found(self, mock_input, mock_load, sample_dataframe, capsys):
        mock_load.return_value = sample_dataframe
        mock_input.side_effect = ['2', 'chocolate', '4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'Chocolate Cake' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_search_by_name_not_found(self, mock_input, mock_load, sample_dataframe, capsys):
        mock_load.return_value = sample_dataframe
        mock_input.side_effect = ['2', 'pizza', '4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'No recipes found' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_search_by_name_empty(self, mock_input, mock_load, sample_dataframe, capsys):
        mock_load.return_value = sample_dataframe
        mock_input.side_effect = ['2', '', '4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'Please enter a search term' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_search_by_name_multiple_versions(self, mock_input, mock_load, capsys):
        df = pd.DataFrame({
            'name': ['Chocolate Cake', 'Chocolate Cake'],
            'minutes': [45, 50],
            'n_steps': [10, 12],
            'n_ingredients': [8, 9],
            'ingredients': ["['flour', 'sugar']", "['flour', 'sugar', 'cocoa']"],
            'steps': ["['Mix', 'Bake']", "['Mix', 'Bake', 'Cool']"],
            'description': ['Version 1', 'Version 2']
        })
        mock_load.return_value = df
        mock_input.side_effect = ['2', 'chocolate', '4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'VERSION' in captured.out
        assert '2 versions found' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_search_by_ingredient_found(self, mock_input, mock_load, sample_dataframe, capsys):
        mock_load.return_value = sample_dataframe
        mock_input.side_effect = ['3', 'chicken', '4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'Chicken Soup' in captured.out
        assert 'ALL MATCHING RECIPES' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_search_by_ingredient_multiple(self, mock_input, mock_load, sample_dataframe, capsys):
        mock_load.return_value = sample_dataframe
        mock_input.side_effect = ['3', 'flour, sugar', '4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'containing ALL of' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_search_by_ingredient_not_found(self, mock_input, mock_load, sample_dataframe, capsys):
        mock_load.return_value = sample_dataframe
        mock_input.side_effect = ['3', 'bacon', '4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'No recipes found' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_search_by_ingredient_empty(self, mock_input, mock_load, sample_dataframe, capsys):
        mock_load.return_value = sample_dataframe
        mock_input.side_effect = ['3', '', '4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'Please enter a search term' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_quit(self, mock_input, mock_load, sample_dataframe, capsys):
        mock_load.return_value = sample_dataframe
        mock_input.side_effect = ['4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'Thank you for using Recipeasy!' in captured.out
    
    @patch('Recipeasy.load_recipe_data')
    @patch('builtins.input')
    def test_main_invalid_choice(self, mock_input, mock_load, sample_dataframe, capsys):
        mock_load.return_value = sample_dataframe
        mock_input.side_effect = ['9', '4']
        
        main()
        captured = capsys.readouterr()
        
        assert 'Invalid choice' in captured.out