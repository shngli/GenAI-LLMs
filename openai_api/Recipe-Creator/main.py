"""
We will generate a recipe and ask the user for the input. Once we have the recipe, 
we pass in the recipe text to Dish2Image() and send that to Dalle.

generate_image() will extract the recipe title to generate the food image, and then 
we store the recipe as a text file. 
"""

from recipe import RecipeGenerator
from dalle import Dish2Image

### Create an instance of the RecipeGenerator class ###
gen = RecipeGenerator()

###  Ask the user for input and create the recipe ###
recipe = gen.generate_recipe()

### Print the recipe ###
print(recipe)

### Create an instance of the Dish2Image class ###
dalle = Dish2Image(recipe)

### Visualize the dish ###
dalle.generate_image()

### Store the recipe in a text file ###
gen.store_recipe(recipe, f"{dalle.title}.txt")