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

locals {
  # deploy/state.py appends a trailing slash if one is missing, so a prefix of
  # "x" means "x/" at runtime — while every helper command and key built here
  # would say "xstate/...". The two would disagree about where the database is,
  # which is a miserable thing to debug for a missing character. Normalized once,
  # in the same shape the runtime uses.
  state_prefix = var.state_prefix == "" ? "" : (
    endswith(var.state_prefix, "/") ? var.state_prefix : "${var.state_prefix}/"
  )
}
