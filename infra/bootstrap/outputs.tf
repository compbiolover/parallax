# Copy these into infra/versions.tf's backend block. They are printed rather
# than wired automatically because a backend configuration cannot be computed —
# Terraform reads it before it evaluates anything.
output "backend_config" {
  description = "The backend block the main config needs."
  value       = <<-EOT

    terraform {
      backend "s3" {
        bucket         = "${aws_s3_bucket.state.id}"
        key            = "parallax/terraform.tfstate"
        region         = "${var.region}"
        dynamodb_table = "${aws_dynamodb_table.locks.name}"
        encrypt        = true
      }
    }
  EOT
}

output "state_bucket" {
  value = aws_s3_bucket.state.id
}

output "lock_table" {
  value = aws_dynamodb_table.locks.name
}
