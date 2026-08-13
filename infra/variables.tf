variable "region" {
  description = "AWS region. Pins ECR, S3 and the schedule; awkward to change later."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Prefix for every resource name."
  type        = string
  default     = "parallax"
}

variable "data_bucket_name" {
  description = <<-EOT
    Globally unique S3 bucket for the run's own state — the SQLite database, the
    eMFD lexicon, and the exported dashboard payload. Separate from the Terraform
    state bucket: different lifecycle, different blast radius.
  EOT
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.data_bucket_name))
    error_message = "Must be a valid S3 bucket name: lowercase, 3-63 chars, no underscores."
  }
}

variable "alarm_email" {
  description = <<-EOT
    Where failure and budget alarms go. The SNS subscription needs confirming
    from this inbox — Terraform will show it as `pending_confirmation` until you
    click the link, and that is not a bug.
  EOT
  type        = string

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.alarm_email))
    error_message = "Must look like an email address."
  }
}

variable "schedule_expression" {
  description = <<-EOT
    EventBridge Scheduler expression. Interpreted in `schedule_timezone`, so this
    is local wall-clock time and DST is handled for you.

    Deliberately not near 00:00: the snapshot date is taken when the step runs,
    so a run straddling UTC midnight files under the next day and leaves a gap.
  EOT
  type        = string
  default     = "cron(0 6 * * ? *)"
}

variable "schedule_timezone" {
  description = "IANA timezone for the schedule, e.g. America/New_York."
  type        = string
  default     = "America/New_York"
}

variable "schedule_enabled" {
  description = <<-EOT
    Whether the daily schedule fires. Start `false`: trigger the state machine by
    hand and watch it once before letting it run unattended.
  EOT
  type        = bool
  default     = false
}

variable "ingest_cpu" {
  description = "Fargate CPU units for the ingest task (4096 = 4 vCPU)."
  type        = number
  default     = 4096
}

variable "ingest_memory" {
  description = "Fargate memory (MiB) for the ingest task. Must pair with cpu."
  type        = number
  default     = 16384
}

variable "aggregate_cpu" {
  description = "Fargate CPU units for the aggregate task."
  type        = number
  default     = 1024
}

variable "aggregate_memory" {
  description = "Fargate memory (MiB) for the aggregate task."
  type        = number
  default     = 2048
}

variable "ingest_timeout_seconds" {
  description = <<-EOT
    Hard ceiling on the ingest step, so a hung run cannot bill overnight.

    Must stay comfortably above 3600: `scoring/liberty.py` polls the Anthropic
    Batch API synchronously for up to an hour, so anything under ~90 minutes
    would kill legitimate runs rather than runaway ones.
  EOT
  type        = number
  default     = 7200

  validation {
    condition     = var.ingest_timeout_seconds >= 5400
    error_message = "Must be >= 5400s. Liberty tagging polls the Batch API for up to 3600s."
  }
}

variable "aggregate_timeout_seconds" {
  description = "Hard ceiling on the aggregate step."
  type        = number
  default     = 3600
}

variable "log_retention_days" {
  description = "CloudWatch log retention. The default is never-expire, i.e. a slow bill."
  type        = number
  default     = 30
}

variable "budget_warn_usd" {
  description = "Monthly spend that triggers a warning email."
  type        = number
  default     = 50
}

variable "budget_alarm_usd" {
  description = "Monthly spend that triggers an alarm email."
  type        = number
  default     = 150
}

variable "github_repository" {
  description = "owner/repo allowed to assume the ECR push role via OIDC."
  type        = string
  default     = "compbiolover/parallax"
}

variable "state_prefix" {
  description = "Key prefix inside the data bucket. Must match PARALLAX_STATE_PREFIX."
  type        = string
  default     = "parallax/"
}
