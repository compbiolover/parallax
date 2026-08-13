terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }

  # Filled in from `terraform output backend_config` in infra/bootstrap.
  # Commented out so `terraform init -backend=false && terraform validate` works
  # in CI with no credentials and no bucket. Uncomment before the first apply.
  #
  # backend "s3" {
  #   bucket         = "REPLACE-ME"
  #   key            = "parallax/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "parallax-tfstate-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "parallax"
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}
