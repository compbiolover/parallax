output "ecr_repositories" {
  description = "Push targets, one per image target in docker/Dockerfile."
  value       = { for k, r in aws_ecr_repository.image : k => r.repository_url }
}

output "github_actions_role_arn" {
  description = "Set as AWS_ROLE_ARN in the repository's Actions variables."
  value       = aws_iam_role.github_ecr_push.arn
}

output "data_bucket" {
  value = aws_s3_bucket.data.id
}

output "lexicon_upload_command" {
  description = "The eMFD lexicon is gitignored and not in the image; it comes from here."
  value       = "aws s3 cp data/emfd_scoring.csv s3://${aws_s3_bucket.data.id}/${local.state_prefix}lexicon/emfd_scoring.csv"
}

output "database_download_command" {
  description = "Pull the run's database down to inspect it with `make history`."
  value       = "aws s3 cp s3://${aws_s3_bucket.data.id}/${local.state_prefix}state/parallax.sqlite ./data/aws-parallax.sqlite"
}

output "secret_set_command" {
  description = "Values are set out of band so they never enter Terraform state."
  value       = "aws secretsmanager put-secret-value --secret-id ${aws_secretsmanager_secret.runtime.name} --secret-string file://runtime.json"
}

output "state_machine_arn" {
  value = aws_sfn_state_machine.daily.arn
}

output "trigger_command" {
  description = "Run it once by hand, and watch, before enabling the schedule."
  value       = "aws stepfunctions start-execution --state-machine-arn ${aws_sfn_state_machine.daily.arn}"
}

output "schedule_state" {
  description = "Reminder: the schedule ships disabled until you flip schedule_enabled."
  value       = "${aws_scheduler_schedule.daily.name} is ${var.schedule_enabled ? "ENABLED" : "DISABLED"} (${var.schedule_expression} ${var.schedule_timezone})"
}
