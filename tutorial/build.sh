
#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The arguments to this script are the image name and application name
image=$1
app=$2

chmod +x $app/train
chmod +x $app/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration
region=$(aws configure get region)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"


# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi


# Edit ECR policy permission rights
aws ecr set-repository-policy --repository-name "${image}" --policy-text ecr_policy.json

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build  -t ${image} --build-arg APP=$app .
docker tag ${image} ${fullname}
