import pandas as pd
import random
import os
import json

def download_dataset():

    try:
        import kaggle
        # Download the dataset
        kaggle.api.dataset_download_files(
            'shuyangli94/food-com-recipes-and-user-interactions',
            path='./data',
            unzip=True
        )
        return True
    
    except ImportError:
        print("Error, please try: pip install kaggle")
        return False
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Installed kaggle: pip install kaggle")
        print("2. Set up Kaggle API credentials (kaggle.json in ~/.kaggle/)")
        print("3. Accepted the dataset terms on Kaggle website")
        return False
        
        """ 
        To set up the API Key whenever you have it saved, follow these steps!

        mkdir -p /home/codespace/.config/kaggle
        nano /home/codespace/.config/kaggle/kaggle.json

        {"username":"your_kaggle_username","key":"your_api_key"}
        """

def load_recipe_data():

    data_paths = [
        './data/RAW_recipes.csv',
        './data/recipes.csv',
        'RAW_recipes.csv',
        'recipes.csv'
    ]
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"Loaded {len(df)} recipes successfully!\n")
                return df
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
    
    if df is None:
        print("Dataset not found locally.")
        user_input = input("Would you like to download it from Kaggle? (yes/no): ").strip().lower()
        
        if user_input == 'yes':
            if download_dataset():
                return load_recipe_data()
        else:
            print("\nPlease download the dataset manually:")
            print("1. Go to Kaggle: https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions")
            print("2. Download the dataset")
            print("3. Extract RAW_recipes.csv to the same folder as this script")
    
    return None

def search_recipes_by_name(df, query):

    query = query.lower()
    matches = df[df['name'].str.lower().str.contains(query, na=False)]
    return matches

def search_recipes_by_ingredient(df, query):

    if 'ingredients' not in df.columns:
        return pd.DataFrame()

    ingredients = [ing.strip().lower() for ing in query.split(',') if ing.strip()]
    
    if not ingredients:
        return pd.DataFrame()
    
    matches = df
    
    for ingredient in ingredients:
        matches = matches[matches['ingredients'].astype(str).str.lower().str.contains(ingredient, na=False, regex=False)]
    
    return matches

def display_recipe(recipe):

    print("\n" + "="*60)
    print("RECIPE RECOMMENDATION")
    print("="*60)
    
    print(f"\nName: {recipe['name']}")
    
    if 'minutes' in recipe.index:
        print(f"Cooking Time: {recipe['minutes']} minutes")
    
    if 'n_steps' in recipe.index:
        print(f"Number of Steps: {recipe['n_steps']}")
    
    if 'n_ingredients' in recipe.index:
        print(f"Number of Ingredients: {recipe['n_ingredients']}")
    
    if 'ingredients' in recipe.index and pd.notna(recipe['ingredients']):
        print(f"\nIngredients:\n{recipe['ingredients']}")
    
    if 'steps' in recipe.index and pd.notna(recipe['steps']):
        print(f"\nInstructions:\n{recipe['steps']}")
    
    if 'description' in recipe.index and pd.notna(recipe['description']):
        print(f"\nDescription:\n{recipe['description']}")
    
    print("\n" + "="*60 + "\n")

def main():

    print("="*60)
    print("RANDOM RECIPE RECOMMENDER")
    print("="*60)
    print("\nWelcome! This program recommends random recipes from Kaggle's")
    print("Recipe Dataset (over 2M recipes).\n")
    
    df = load_recipe_data()
    
    if df is None:
        return
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Get a completely random recipe")
        print("2. Search for recipes by name")
        print("3. Search for recipes by ingredient")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Get completely random recipe
            random_recipe = df.sample(n=1).iloc[0]
            display_recipe(random_recipe)
            
        elif choice == '2':
            # Search by name
            query = input("\nEnter recipe name to search: ").strip()
            
            if not query:
                print("Please enter a search term.")
                continue
            
            matches = search_recipes_by_name(df, query)
            
            if len(matches) == 0:
                print(f"\nNo recipes found with name matching '{query}'. Try another search!")
            else:
                print(f"\nFound {len(matches)} recipes with name matching '{query}'!")
                
                # Check if there are multiple recipes with the same exact name
                for name in matches['name'].unique():
                    versions = matches[matches['name'] == name]
                    
                    if len(versions) > 1:
                        # Multiple versions of the same recipe
                        print("\n" + "="*60)
                        print(f"{name} ({len(versions)} versions found):")
                        print("="*60)
                        
                        for idx, (_, recipe) in enumerate(versions.iterrows(), 1):
                            print(f"\n  VERSION {idx}:")
                            print(f"Name: {recipe['name']}")
                            
                            if 'minutes' in recipe.index:
                                print(f"Cooking Time: {recipe['minutes']} minutes")
                            
                            if 'n_steps' in recipe.index:
                                print(f"Number of Steps: {recipe['n_steps']}")
                            
                            if 'n_ingredients' in recipe.index:
                                print(f"Number of Ingredients: {recipe['n_ingredients']}")
                            
                            if 'ingredients' in recipe.index and pd.notna(recipe['ingredients']):
                                print(f"Ingredients: {recipe['ingredients']}")
                            
                            if 'steps' in recipe.index and pd.notna(recipe['steps']):
                                print(f"Instructions: {recipe['steps']}")
                            
                            if 'description' in recipe.index and pd.notna(recipe['description']):
                                print(f"Description: {recipe['description']}")
                            
                            print("-" * 60)
                    else:
                        # Only one version, display normally
                        random_recipe = versions.iloc[0]
                        display_recipe(random_recipe)
            
        elif choice == '3':
            # Search by ingredient
            print("\nEnter ingredient(s) to search:")
            print("(For multiple ingredients, separate with commas. Example: chicken, garlic, tomato)")
            query = input("Ingredient(s): ").strip()
            
            if not query:
                print("Please enter a search term.")
                continue
            
            matches = search_recipes_by_ingredient(df, query)
            
            if len(matches) == 0:
                print(f"\nNo recipes found with ingredient(s) '{query}'. Try another search!")
            else:
                ingredients_list = [ing.strip() for ing in query.split(',')]
                if len(ingredients_list) > 1:
                    print(f"\nFound {len(matches)} recipes containing ALL of: {', '.join(ingredients_list)}")
                else:
                    print(f"\nFound {len(matches)} recipes with ingredient '{query}'!")
                
                print("\n" + "="*60)
                print("ALL MATCHING RECIPES:")
                print("="*60)
                for idx, recipe_name in enumerate(matches['name'], 1):
                    print(f"{idx}. {recipe_name}")
                print("="*60)
                
                print("\nDisplaying a random recipe from the results:")
                random_recipe = matches.sample(n=1).iloc[0]
                display_recipe(random_recipe)

        elif choice == '4':
            print("\nThank you for using Recipe Recommender!")
            break
        
        else:
            print("\nInvalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
