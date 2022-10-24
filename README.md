Project Title: L3T11

Description: This image contains the means of running garden.py which has some file requirements such as spaCy.

Installation: The instructions on how to run this dockerfile are specified below:

DOCKER FILE:
1.  Navigate to https://hub.docker.com/ and create a Docker Hub account;
2.  Download Docker desktop and install from https://www.docker.com/products/docker-desktop then login;
3.  Navigate to the folder with the Dockerfile using cd;
4.  Execute the two commands in commands.txt in terminal, which are:
    docker build --tag my-project .  # This creates the docker image and calls it my Project
    docker run my-project  # This "containerises" the image, and runs it via python 3 in the command line
