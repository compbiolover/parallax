resource "aws_sns_topic" "alarms" {
  name = "${var.name}-alarms"
}

# Needs confirming from the inbox. Terraform will show this as
# `pending_confirmation` until you click the link, and that is not a failure —
# it cannot be confirmed programmatically, by design.
resource "aws_sns_topic_subscription" "alarms_email" {
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# Belt to the state machine's braces: the Failed state publishes to SNS itself,
# but that only fires if the execution reached it. An execution that is killed
# outright — a timeout, a quota — never runs its own catch block.
resource "aws_cloudwatch_metric_alarm" "executions_failed" {
  alarm_name          = "${var.name}-daily-failed"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  threshold           = 1
  period              = 3600
  namespace           = "AWS/States"
  metric_name         = "ExecutionsFailed"
  statistic           = "Sum"
  treat_missing_data  = "notBreaching"

  dimensions = {
    StateMachineArn = aws_sfn_state_machine.daily.arn
  }

  alarm_description = "The Parallax daily state machine recorded a failed execution."
  alarm_actions     = [aws_sns_topic.alarms.arn]
}

# Catches the opposite failure: nothing ran at all. A disabled schedule, a
# scheduler that cannot assume its role, or an account-level problem produces
# silence, and silence looks exactly like success from the outside.
resource "aws_cloudwatch_metric_alarm" "executions_missing" {
  alarm_name          = "${var.name}-daily-did-not-run"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  threshold           = 1
  period              = 129600 # 36h — one missed day, not one late morning
  namespace           = "AWS/States"
  metric_name         = "ExecutionsStarted"
  statistic           = "Sum"
  treat_missing_data  = "breaching"

  dimensions = {
    StateMachineArn = aws_sfn_state_machine.daily.arn
  }

  alarm_description = "No Parallax execution started in 36 hours. The run is silently not happening."
  alarm_actions     = [aws_sns_topic.alarms.arn]
}

# A new account running an unattended scheduled job has no natural ceiling. AWS
# Budgets is free for the first two, and the two plausible runaways here are a
# hung task and one that restarts.
resource "aws_budgets_budget" "monthly" {
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_alarm_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Absolute, not a percentage of the limit. Expressed as a percentage this is
  # `50 / 150 * 100`, which Terraform evaluates to 33.333…  to a hundred decimal
  # places — a value AWS normalises differently, so the plan never converges and
  # shows a diff on every apply.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = var.budget_warn_usd
    threshold_type             = "ABSOLUTE_VALUE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alarm_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alarm_email]
  }

  # The one that arrives in time to do something about it.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alarm_email]
  }
}
