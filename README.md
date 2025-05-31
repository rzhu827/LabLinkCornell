# LabLink Cornell

A Flask-based web application for semantically exploring Cornell professors' research profiles. Users can search by interest and filter by citation count.

## 🧠 Overview
This project was built as part of Cornell's CS/INFO 4300 course. It leverages Flask for backend routing, a lightweight JSON backend as a database alternative, and custom frontend styling for a clean and intuitive search experience.
- Users can query topics of interest and view matched professors.
- Citation filters, semantic profiling, similar professors, and Google Scholar links help refine exploration.
- Designed with Vanta background animation and responsive layout.

### See below for more information.


## 📦 Contents

- [🚀 Deploying on the server](#-deploying-on-the-server)
- [💻 Running Locally](#-running-locally)
- [📁 Uploading Large Files](#-uploading-large-files)
- [🐞 Debugging Some Basic Errors](#-debugging-some-basic-errors)
- [🧪 Virtual Environments and Dependency Tracking](#-virtual-environments-and-dependency-tracking)
- [🛠️ Troubleshooting](#-troubleshooting)
- [📝 General Comments](#-general-comments)

## 🚀 Deploying on the server

### Step 0: Forking or Cloning this template

- You should make a FORK or CLONE of this repository on (regular/your public) GitHub, make sure that your repository is PUBLIC.

### Step 1: Login to the deployment dashboard

- Login to the dashboard at https://4300showcase.infosci.cornell.edu/login using your provided account name and password. Each team was provided with joint account in CMS comments to assignment P01; since you are sharing an account, any changes made by one teammate will reflect for everyone in the team.
- When you login, ensure your dashboard has the following data filled from the image below (check the black arrows only)
  - The GitHub URL field will not be filled in for you, so you should add in the URL of your forked repository.

![image](https://github.com/CornellNLP/4300-Flask-Template-JSON/assets/42887079/4c4c8004-c82f-4660-9af1-0cc898721013)


### Step 2: Understanding the interface

- **CLONE**: First time clone from GitHub onto the server, this is to load your files on the server. In future, when you push updates, clone will re-downloaded your new files onto the server. It is imperative that you re-clone before building.
- **BUILD**: Will re-clone and build everything from your GitHub repo, and only from the master/main branch. This is a hard reset, however, your data will be preserved. This includes all data from your database and tables. 
- **DESTROY**: Will destroy all your containers as well as remove any data associated with them. Useful for fresh boot from scratch
- **Container Information Table**: Will show you the status of all your containers. This should tell you if they are on/off. Generally, this information is just useful for debugging and for checking any port errors or mismatches, although mostly just useful for TAs.
- **Logs**: Should give you an idea of what went wrong during deployment. This of course will not tell you if something is broken during build time, but only what happened when your code was deployed. 

### Step 3: Test deployment

- On the dashboard, in the provided search bar, add the URL of your forked repository
- Click the **clone** button and wait for a bit till you get a confirmation
- Click **build**, and wait for a minute. If all goes successfully, hitting the refresh button on the Container Information table and the logs tab will show the created data from the service. If this doesn't work, logout and log back in.
- Now, clicking on the URL button should lead you to a simple episode-searching app
- If it doesn't load initially, give it a few seconds and reload.

## 💻 Running locally

This is to help you test and develop your app locally. Ensure that you have Python version 3.10 or above installed on your machine (ideally in a virtual environment). Some of the libraries and code used in the template, as well as on the server end, are only compatible with Python versions 3.10 and above.
  
### Step 1: Set up a virtual environment
Create a virtual environment in Python. You may continue using the one you setup for assignment if necessary. To review how to set up a virtual environment and activate it, refer to A0 assignment writeup.

Run `python -m venv <virtual_env_name>` in your project directory to create a new virtual environment, remember to change <virtual_env_name> to your preferred environment name.

### Step 2: Install dependencies
You need to install dependencies by running `python -m pip install -r requirements.txt` in the backend folder.

### Step 3: Modify init.json file
This project gives you an init.json file with dummy data to see how app.py file reads data from the json file. 
You can change data in this file to your project's json data, but do not delete or change the name of the file. However, you are allowed to create more json files for your project. 

## Command to run project locally: 
```flask run --host=0.0.0.0 --port=5000```

## 📁 Uploading Large Files 
- Note: This feature is correctly under testing
- When your dataset is ready, it should be of the form of a JSON file of 128MB or less.
  - 128MB is negotiable, based on your dataset requirements
- Click "Upload JSON file" button, choose your file and hit the upload button to send it to your project
- The files are chunked. Any interruption either on the network or client end will require a full file re-upload so be careful
  - In the event your file does not get consistently uploaded due to network issues or takes too long (it really shouldn't) you may request a manual upload
- This JSON file that you upload will always replace your **init.json** file. This means that when you build your project, this file will be automatically imported into your Database and be available to use.

## 🐞 Debugging Some Basic Errors
- After the build, wait a few seconds as the server will still be loading, especially for larger applications with a lot of setup
- **Do not change the Dockerfiles without permission**
- Sometimes, if a deployment doesn't work, you can try logging out and back in to see if it works
- Alternatively, checking the console will tell you what error it is. If it's a 401, then logging in and out should fix it. 
- If it isn't a 401, first try checking the logs or container status. Check if the containers are alive or not, which could cause issues. If the containers are down, try stopping and starting them. If that does not work, you can report it on ED.
- If data isn't important, destroying and then cloning and re-building containers will usually fix the issue (assuming there's no logical error)

## 🧪 Virtual Environments and Dependency Tracking
- It's essential to avoid uploading your virtual environments, as they can significantly inflate the size of your project. Large repositories will lead to issues during cloning, especially when memory limits are crossed (Limit – 2GB). 
To prevent your virtual environment from being tracked and uploaded to GitHub, follow these steps:
1. **Exclude Virtual Environment**
   - Navigate to your project's root directory and locate the `.gitignore` file. 
   - Add the name of your virtual environment directory to this file in the following format: `<virtual_environment_name>/`. This step ensures that Git ignores the virtual environment folder during commits.

2. **Remove Previously Committed Virtual Environment**
   - If you've already committed your virtual environment to the repository, you can remove it from the remote repository by using Git commands to untrack and delete it. You will find resources online to do so.
Afterward, ensure to follow step 1 to prevent future tracking of virtual environment.

3. **Managing Dependencies**
    - Add all the new libraries you downloaded using pip install for your project to the existing `requirements.txt` file. To do so,
    - Navigate to your project backend directory and run the command `pip freeze > requirements.txt`. This command will create or overwrite the `requirements.txt` file with a list of installed packages and their versions. 
    - Our server will use your project’s `requirements.txt` file to install all required packages, ensuring that your project runs seamlessly.

## 🛠 Troubleshooting

The attached google document includes a compilation of frequent issues encountered by students across various project stages, detailing whether these issues have been resolved and the solutions that were effective. We will continue to update this list with new information.

Link: https://docs.google.com/document/d/1sF2zsubii_SYJLfZN02UB9FvtH1iLmi9xd-X4wbpbo8

## 📝 General Comments

This project was developed by:
Beau Hayes-Pollard, Analeah Real, Joice Chen, Tina Chen, and Richard Zhu

README.md adapted from original authors.
