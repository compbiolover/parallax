# The chicken and the egg: Terraform's own state has to live somewhere, and that
# somewhere cannot itself be managed by the state it holds. So this one small
# config runs with *local* state, creates the bucket and lock table, and is then
# essentially never touched again.
#
#   cd infra/bootstrap && terraform init && terraform apply
#
# Its terraform.tfstate is gitignored and describes two resources. Losing it is
# survivable — `terraform import` recovers it, or you leave the two resources
# alone forever, which is what happens in practice.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "parallax"
      ManagedBy = "terraform"
      Component = "bootstrap"
    }
  }
}

resource "aws_s3_bucket" "state" {
  bucket = var.state_bucket_name

  # State is the one thing whose loss is unrecoverable — it is how Terraform
  # knows what it already created. Deleting it should take deliberate effort.
  lifecycle {
    prevent_destroy = true
  }
}

# Versioning is not a nicety here. A corrupted or truncated state file is
# recoverable only if the previous version still exists.
resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# State holds every value an apply read or generated, in plaintext. It is a
# secret store that happens to look like a build artifact.
resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Old state versions accumulate one per apply forever otherwise.
resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    id     = "expire-old-state-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [aws_s3_bucket_versioning.state]
}

# Stops two applies from writing state simultaneously. One operator today, but
# the failure it prevents — two concurrent applies interleaving writes — leaves
# state describing infrastructure that never existed.
resource "aws_dynamodb_table" "locks" {
  name         = var.lock_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  lifecycle {
    prevent_destroy = true
  }
}
