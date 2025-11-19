import os
import kagglehub
import sqlite3
import tkinter as tk
path = kagglehub.dataset_download("wilmerarltstrmberg/recipe-dataset-over-2m")

'''
class RecipeModel:
    def __init__(self, db_path):
        self.db_path = db_path

    def fetch_recipes(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM recipes LIMIT 10;")
        recipes = cursor.fetchall()
        conn.close()
        return recipes

class RecipeView:
    def display_recipes(self, recipes):
        print("Recipes:")
        for r in recipes:
            print(f"{r[0]}: {r[1]}")

class RecipeController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def list_recipes(self):
        recipes = self.model.fetch_recipes()
        self.view.display_recipes(recipes)
'''
class Model:
    def __init__(self):
        self.text = "Hello from the Model!"

    def get_text(self):
        return self.text

class View(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.message = tk.StringVar()
        self.label = tk.Label(self, textvariable=self.message)
        self.label.pack()
        self.show_button = tk.Button(self, text="Show Message")
        self.show_button.pack()

    def set_controller(self, controller):
        self.show_button.config(command=controller.show_message)

    def set_message(self, text):
        self.message.set(text)


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.set_controller(self)

    def show_message(self):
        message = self.model.get_text()
        self.view.set_message(message)

if __name__ == "__main__":
    root = tk.Tk()
    model = Model()
    view = View(master=root)
    controller = Controller(model, view)
    view.mainloop()
