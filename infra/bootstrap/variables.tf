variable "region" {
  description = "AWS region. Pins where state lives; the main config uses the same one."
  type        = string
  default     = "us-east-1"
}

variable "state_bucket_name" {
  description = <<-EOT
    Globally unique S3 bucket name for Terraform state. Bucket names are shared
    across every AWS account on earth, so this needs a suffix nobody else has —
    an account id or a random word, not just "parallax-tfstate".
  EOT
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.state_bucket_name))
    error_message = "Must be a valid S3 bucket name: lowercase, 3-63 chars, no underscores."
  }
}

variable "lock_table_name" {
  description = "DynamoDB table name for state locking."
  type        = string
  default     = "parallax-tfstate-locks"
}
