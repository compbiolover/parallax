# Two roles, and conflating them is the classic way this stack fails.
#
# The **execution role** is what ECS itself uses, before any of our code runs:
# pulling the image, fetching secrets, opening a log stream. The **task role** is
# what the process inside the container uses. Grant S3 to the execution role and
# the task starts cleanly and then cannot read its own database — an error that
# arrives late and points nowhere useful.

# ---------------------------------------------------------------- execution ---

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

# ECR pull plus CloudWatch Logs.
resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Secrets injection happens in the execution role, not the task role: ECS reads
# the secret and hands the container an environment variable, so the container
# never needs Secrets Manager access itself.
data "aws_iam_policy_document" "execution_secrets" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.runtime.arn]
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  name   = "${var.name}-execution-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets.json
}

# --------------------------------------------------------------------- task ---

resource "aws_iam_role" "task" {
  name               = "${var.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "task" {
  # Exactly what deploy/state.py's pull and push call.
  statement {
    sid = "StateObjects"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      # Not optional, and the reason is easy to miss: boto3's upload_file
      # switches to a multipart upload above 8 MB, and the database is hundreds
      # of megabytes. A GetObject/PutObject-only policy passes every test
      # against a small fixture and fails against the real store, with an
      # AccessDenied that names an operation nobody granted because nobody
      # realised it would be used.
      "s3:AbortMultipartUpload",
    ]
    resources = ["${aws_s3_bucket.data.arn}/${var.state_prefix}*"]
  }

  statement {
    sid       = "ListForMultipart"
    actions   = ["s3:ListBucketMultipartUploads"]
    resources = [aws_s3_bucket.data.arn]
  }

  # Exactly what acquire, release and _describe_lease call — no more.
  statement {
    sid = "Lease"
    actions = [
      "dynamodb:PutItem",
      "dynamodb:DeleteItem",
      "dynamodb:GetItem",
    ]
    resources = [aws_dynamodb_table.lease.arn]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "${var.name}-task"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

# ------------------------------------------------------------ step functions ---

data "aws_iam_policy_document" "states_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "states" {
  name               = "${var.name}-states"
  assume_role_policy = data.aws_iam_policy_document.states_assume.json
}

data "aws_iam_policy_document" "states" {
  statement {
    sid       = "RunTasks"
    actions   = ["ecs:RunTask"]
    resources = [for t in aws_ecs_task_definition.task : "${replace(t.arn, "/:\\d+$/", "")}:*"]
  }

  statement {
    sid       = "StopAndDescribe"
    actions   = ["ecs:StopTask", "ecs:DescribeTasks"]
    resources = ["*"]
  }

  # RunTask.sync needs these to be notified when the task finishes, rather than
  # polling. Without them the state machine hangs until its own timeout.
  statement {
    sid       = "SyncCallback"
    actions   = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
    resources = ["arn:aws:events:${var.region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForECSTaskRule"]
  }

  # Handing the task its roles is itself a permission.
  statement {
    sid       = "PassRoles"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn, aws_iam_role.task.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com"]
    }
  }

  statement {
    sid       = "Notify"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.alarms.arn]
  }
}

resource "aws_iam_role_policy" "states" {
  name   = "${var.name}-states"
  role   = aws_iam_role.states.id
  policy = data.aws_iam_policy_document.states.json
}

# ---------------------------------------------------------------- scheduler ---

data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
    # Stops any other account's scheduler from assuming this role.
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "${var.name}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume.json
}

resource "aws_iam_role_policy" "scheduler" {
  name = "${var.name}-scheduler"
  role = aws_iam_role.scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "states:StartExecution"
      Resource = aws_sfn_state_machine.daily.arn
    }]
  })
}
