locals {
  # Space-separated, and that is not cosmetic: `--only` is nargs="+" with
  # choices=STEPS, so a comma-joined "ingest,backfill" is one invalid token and
  # argparse exits 2 — indistinguishable from a genuinely failed run.
  #
  # `digest` is named explicitly because it is deliberately absent from
  # DEFAULT_STEPS (daily/runner.py). Leaving it implicit deploys a pipeline that
  # does everything except send the email, and reports success for it.
  commands = {
    ingest    = ["--only", "ingest", "backfill"]
    aggregate = ["--only", "cluster", "summarize", "snapshot", "export", "digest"]
  }

  task_image = {
    ingest    = "scored"
    aggregate = "core"
  }

  task_size = {
    ingest    = { cpu = var.ingest_cpu, memory = var.ingest_memory }
    aggregate = { cpu = var.aggregate_cpu, memory = var.aggregate_memory }
  }

  # Injected into both tasks. Values live in Secrets Manager; these are only the
  # names of the keys to pull out of the one JSON document.
  secret_keys = [
    "ANTHROPIC_API_KEY",
    "PARALLAX_SMTP_HOST",
    "PARALLAX_SMTP_PORT",
    "PARALLAX_SMTP_USER",
    "PARALLAX_SMTP_PASSWORD",
    "PARALLAX_DIGEST_TO",
    "PARALLAX_DIGEST_FROM",
    "PARALLAX_SMTP_STARTTLS",
  ]
}

resource "aws_ecs_cluster" "main" {
  name = var.name

  setting {
    name  = "containerInsights"
    value = "disabled" # a per-metric bill for two tasks a day
  }
}

resource "aws_cloudwatch_log_group" "task" {
  for_each = local.commands

  name              = "/ecs/${var.name}/${each.key}"
  retention_in_days = var.log_retention_days
}

resource "aws_ecs_task_definition" "task" {
  for_each = local.commands

  family                   = "${var.name}-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = local.task_size[each.key].cpu
  memory                   = local.task_size[each.key].memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = each.key
    image     = "${aws_ecr_repository.image[local.task_image[each.key]].repository_url}:latest"
    command   = each.value
    essential = true

    environment = [
      { name = "PARALLAX_STATE_BUCKET", value = aws_s3_bucket.data.id },
      { name = "PARALLAX_STATE_PREFIX", value = local.state_prefix },
      { name = "PARALLAX_LEASE_TABLE", value = aws_dynamodb_table.lease.name },
      { name = "PARALLAX_LEASE_ID", value = "daily" },
      { name = "AWS_DEFAULT_REGION", value = var.region },
      # Refuse to score with the built-in demo lexicon. Without this a missing
      # eMFD file produces a complete, plausible, entirely unvalidated set of
      # numbers — the one failure that looks exactly like success.
      { name = "PARALLAX_REQUIRE_LEXICON", value = each.key == "ingest" ? "1" : "0" },
    ]

    secrets = [
      for key in local.secret_keys : {
        name      = key
        valueFrom = "${aws_secretsmanager_secret.runtime.arn}:${key}::"
      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.task[each.key].name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "run"
      }
    }

    # Bounds the shutdown, not the run — the step timeout in the state machine
    # bounds the run. Both exist because they catch different failures.
    stopTimeout = 120
  }])
}
