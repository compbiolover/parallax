# Two tasks in sequence, because EventBridge Scheduler invokes a single target
# and the two use different images. `RunTask.sync` waits for each to finish,
# rather than the "schedule the second one 90 minutes later and hope" pattern.

locals {
  network_configuration = {
    AwsvpcConfiguration = {
      Subnets        = aws_subnet.public[*].id
      SecurityGroups = [aws_security_group.tasks.id]
      # Load-bearing. Without a public IP and without a NAT gateway, the task
      # cannot reach ECR and dies at image pull with an error that never
      # mentions networking.
      AssignPublicIp = "ENABLED"
    }
  }
}

resource "aws_sfn_state_machine" "daily" {
  name     = "${var.name}-daily"
  role_arn = aws_iam_role.states.arn

  definition = jsonencode({
    Comment = "Parallax daily run: ingest, then aggregate."
    StartAt = "Ingest"
    States = {
      Ingest = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = merge({
          Cluster              = aws_ecs_cluster.main.arn
          TaskDefinition       = aws_ecs_task_definition.task["ingest"].arn
          LaunchType           = "FARGATE"
          NetworkConfiguration = local.network_configuration
        })
        TimeoutSeconds = var.ingest_timeout_seconds

        # Infrastructure faults only. Deliberately *not* States.TaskFailed:
        # daily/__main__.py returns 1 for a partial failure, a total failure and
        # a pre-loop crash alike, and 2 for a malformed command — so a retry
        # here cannot tell "GDELT was down" from "this invocation can never
        # work", and would repeat the latter until the timeout.
        Retry = [{
          ErrorEquals     = ["ECS.AmazonECSException", "ECS.ServerException"]
          IntervalSeconds = 30
          MaxAttempts     = 2
          BackoffRate     = 2
        }]

        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "Failed"
        }]

        Next = "Aggregate"
      }

      Aggregate = {
        Type     = "Task"
        Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = merge({
          Cluster              = aws_ecs_cluster.main.arn
          TaskDefinition       = aws_ecs_task_definition.task["aggregate"].arn
          LaunchType           = "FARGATE"
          NetworkConfiguration = local.network_configuration
        })
        TimeoutSeconds = var.aggregate_timeout_seconds

        Retry = [{
          ErrorEquals     = ["ECS.AmazonECSException", "ECS.ServerException"]
          IntervalSeconds = 30
          MaxAttempts     = 2
          BackoffRate     = 2
        }]

        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "Failed"
        }]

        End = true
      }

      # Notify, then fail the execution — so the run shows as failed in the
      # console rather than as a success that happened to send an email.
      Failed = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn    = aws_sns_topic.alarms.arn
          Subject     = "Parallax daily run failed"
          "Message.$" = "States.JsonToString($)"
        }
        Next = "Fail"
      }

      Fail = {
        Type  = "Fail"
        Cause = "A step of the daily run failed. See the SNS notification and the task logs."
      }
    }
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.states.arn}:*"
    include_execution_data = true
    level                  = "ERROR"
  }
}

resource "aws_cloudwatch_log_group" "states" {
  name              = "/aws/vendedlogs/states/${var.name}-daily"
  retention_in_days = var.log_retention_days
}

# EventBridge *Scheduler*, not a classic EventBridge rule: it takes a named
# timezone and handles DST, so the run stays at the same local hour instead of
# drifting an hour twice a year.
resource "aws_scheduler_schedule" "daily" {
  name       = "${var.name}-daily"
  group_name = "default"
  state      = var.schedule_enabled ? "ENABLED" : "DISABLED"

  schedule_expression          = var.schedule_expression
  schedule_expression_timezone = var.schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_sfn_state_machine.daily.arn
    role_arn = aws_iam_role.scheduler.arn

    retry_policy {
      # The state machine handles its own failure. A scheduler-level retry would
      # start a *second* execution alongside the first, which is exactly what
      # the lease exists to refuse.
      maximum_retry_attempts = 0
    }
  }
}
