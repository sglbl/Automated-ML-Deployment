# ML Ops with GitHub Actions and Azure Machine Learning

This repository is to set up prediction function for data science / machine learning project with automated training and deployment using GitHub Actions and Azure Machine Learning. 

# Getting started

### YouTube Video

Used this video to setup:
[Azure Machine Learning GitHub Actions - Setup Guide](http://www.youtube.com/watch?v=bmFr0LYo_6o "Azure Machine Learning GitHub Actions - Setup Guide")

### 1. Prerequisites

The following prerequisites are required to make this repository work:
- Azure subscription
- Contributor access to the Azure subscription
- Access to GitHub Actions

### 2. Create repository

To get started with ML Ops, simply created a new repo based off this template repo:

https://github.com/machine-learning-apps/ml-template-azure

### 3. Setting up the required secrets

A service principal needs to be generated for authentication and getting access to your Azure subscription. First, I added a service principal with contributor rights to a new resource group or to the one where I've deployed existing Azure Machine Learning workspace. Then I went to the Azure Portal to find the details of your resource group or workspace. Then I've started the Cloud CLI / Azure CLI on computer and executed the following command to generate the required credentials:

```sh
# Replace {service-principal-name}, {subscription-id} and {resource-group} with your 
# Azure subscription id and resource group name and any name for your service principle
az ad sp create-for-rbac --name {service-principal-name} \
                         --role contributor \
                         --scopes /subscriptions/{subscription-id}/resourceGroups/{resource-group} \
                         --sdk-auth
```

This will generate the following JSON output:

```sh
{
  "clientId": "<GUID>",
  "clientSecret": "<GUID>",
  "subscriptionId": "<GUID>",
  "tenantId": "<GUID>",
  (...)
}
```

I added this JSON output as [a secret](https://help.github.com/en/actions/configuring-and-managing-workflows/creating-and-storing-encrypted-secrets#creating-encrypted-secrets) with the name `AZURE_CREDENTIALS` in my GitHub repository:

To do so, Settings tab -> Secrets -> Add the new secret with the name `AZURE_CREDENTIALS` to repository.

### 4. Defining workspace parameters

First I've modified the parameters in the <a href="/.cloud/.azure/workspace.json">`/.cloud/.azure/workspace.json"` file</a> in your repository, so that the GitHub Actions create or connect to the desired Azure Machine Learning workspace.

I used the same value for the `resource_group` parameter that I have used when generating the azure credentials. (If you already have an Azure ML Workspace under that resource group, change the `name` parameter in the JSON file to the name of your workspace, if you want the Action to create a new workspace in that resource group, pick a name for your new workspace, and assign it to the `name` parameter. You can also delete the `name` parameter, if you want the action to use the default value, which is the repository name.)

After saving changes to the file, the predefined GitHub workflow that trains and deploys a model on Azure Machine Learning gets triggered. Check the actions tab to view if your actions have successfully run.

### 5. Modify the code

Then I modified the code in the <a href="/code">`code` folder</a>, so that my model and not the provided sample model gets trained on Azure. (Where required, modify the environment yaml so that the training and deployment environments will have the correct packages installed in the conda environment for your training and deployment.
Upon pushing the changes, actions will kick off your training and deployment run. Check the actions tab to view if your actions have successfully run.)

Comment lines 39 to 55 in <a href="/.github/workflows/train_deploy.yml">`"/.github/workflows/train_deploy.yml"` file</a> if only want to train the model. Uncomment line 7 to 8, if only want to kick off the workflow when pushing changes to the `"/code/"` file.

### 6. Viewing AML resources and runs

The log outputs of your action will provide URLs to view the resources that have been created in AML. Alternatively, visiting the [Machine Learning Studio](https://ml.azure.com/) to view the progress of your runs, etc.

# Documentation

## Code structure

| File/folder                   | Description                                |
| ----------------------------- | ------------------------------------------ |
| `code`                        | Sample data science source code that will be submitted to Azure Machine Learning to train and deploy machine learning models. |
| `code/train`                  | Sample code that is required for training a model on Azure Machine Learning. |
| `code/train/train.py`         | Training script that gets executed on a cluster on Azure Machine Learning. |
| `code/train/environment.yml`  | Conda environment specification, which describes the dependencies of `train.py`. These packages will be installed inside a Docker image on the Azure Machine Learning compute cluster, when executing your `train.py`. |
| `code/train/run_config.yml`   | YAML files, which describes the execution of your training run on Azure Machine Learning. This file also references your `environment.yml`. Please look at the comments in the file for more details. |
| `code/deploy`                 | Sample code that is required for deploying a model on Azure Machine Learning. |
| `code/deploy/score.py`        | Inference script that is used to build a Docker image and that gets executed within the container when you send data to the deployed model on Azure Machine Learning. |
| `code/deploy/environment.yml` | Conda environment specification, which describes the dependencies of `score.py`. These packages will be installed inside the Docker image that will be used for deploying your model. |
| `code/test/test.py`           | Test script that can be used for testing your deployed webservice. Add a `deploy.json` to the `.cloud/.azure` folder and add the following code `{ "test_enabled": true }` to enable tests of your webservice. Change the code according to the tests that zou would like to execute. |
| `.cloud/.azure`               | Configuration files for the Azure Machine Learning GitHub Actions. Please visit the repositories of the respective actions and read the documentation for more details. |
| `.github/workflows`           | Folder for GitHub workflows. The `train_deploy.yml` sample workflow shows you how your can use the Azure Machine Learning GitHub Actions to automate the machine learning process. |
| `docs`                        | Resources for this README.                 |
| `CODE_OF_CONDUCT.md`          | Microsoft Open Source Code of Conduct.     |
| `LICENSE`                     | The license for the sample.                |
| `README.md`                   | This README file.                          |
| `SECURITY.md`                 | Microsoft Security README.                 |

## Documentation of Azure Machine Learning GitHub Actions

The template uses the open source Azure certified Actions listed below. Click on the links and read the README files for more details.
- [aml-workspace](https://github.com/Azure/aml-workspace) - Connects to or creates a new workspace
- [aml-compute](https://github.com/Azure/aml-compute) - Connects to or creates a new compute target in Azure Machine Learning
- [aml-run](https://github.com/Azure/aml-run) - Submits a ScriptRun, an Estimator or a Pipeline to Azure Machine Learning
- [aml-registermodel](https://github.com/Azure/aml-registermodel) - Registers a model to Azure Machine Learning
- [aml-deploy](https://github.com/Azure/aml-deploy) - Deploys a model and creates an endpoint for the model

# What is MLOps?

MLOps empowers data scientists and machine learning engineers to bring together their knowledge and skills to simplify the process of going from model development to release/deployment. ML Ops enables you to track, version, test, certify and reuse assets in every part of the machine learning lifecycle and provides orchestration services to streamline managing this lifecycle. This allows practitioners to automate the end to end machine Learning lifecycle to frequently update models, test new models, and continuously roll out new ML models alongside your other applications and services.

This repository enables Data Scientists to focus on the training and deployment code of their machine learning project (`code` folder of this repository). Once new code is checked into the `code` folder of the master branch of this repository the GitHub workflow is triggered and open source Azure Machine Learning actions are used to automatically manage the training through to deployment phases.
