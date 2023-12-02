ACR_NAME=sarthak output
DEP_IMAGE_NAME=file image
FINAL_IMAGE_NAME=final-image
TAG=latest
az acr login --name $ACR_NAME
docker build -t $ACR_NAME.azurecr.io/$DEP_IMAGE_NAME:$TAG -f DependencyDockerfile .
docker push $ACR_NAME.azurecr.io/$DEP_IMAGE_NAME:$TAG
docker build -t $ACR_NAME.azurecr.io/$FINAL_IMAGE_NAME:$TAG -f FinalDockerfile .
docker push $ACR_NAME.azurecr.io/$FINAL_IMAGE_NAME:$TAG
