# Recipes to Images
Goal: Use a conditional GAN to create images of recipes

## Resources
- Recipe api to query https://developer.edamam.com/edamam-recipe-api-demo
- Deploying with kubernetes: https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app

## TODO:
- [X] Write API query-er
- [X] Get NYT ingredient parser working, process ingredient data
- [X] Run data collection, store in GCP
- [X] Build conditional GAN to create images based on ingredients
- [X] Deploy model with flask, docker, and kubernetes
