# One secret holding eight keys, not eight secrets.
#
# Secrets Manager bills per secret per month, and ECS `valueFrom` can select a
# key out of a JSON document — so this costs $0.40 rather than $3.20 and rotates
# as a single unit.
#
# Terraform creates the *container* and never the values. Anything Terraform
# knows is in state in plaintext; the point of a secret store is that this one
# thing is not. Set them after the first apply:
#
#   aws secretsmanager put-secret-value --secret-id parallax/runtime \
#     --secret-string file://runtime.json
#
# Keys, matching what the code reads from the environment:
#   ANTHROPIC_API_KEY        scoring/claude_client.py
#   PARALLAX_SMTP_HOST       digest/send.py — all four or it declines to send
#   PARALLAX_SMTP_PORT
#   PARALLAX_SMTP_USER
#   PARALLAX_SMTP_PASSWORD
#   PARALLAX_DIGEST_TO       the +parallax-aws alias while running in parallel
#   PARALLAX_DIGEST_FROM
#   PARALLAX_SMTP_STARTTLS

resource "aws_secretsmanager_secret" "runtime" {
  name        = "${var.name}/runtime"
  description = "Runtime environment for the Parallax daily run. Values set out of band."

  # Long enough to undo a mistake, short enough that a rotated credential really
  # goes away.
  recovery_window_in_days = 7
}

# Deliberately no aws_secretsmanager_secret_version resource. Adding one would
# put every value into terraform.tfstate in plaintext, which is the thing the
# .gitignore rules exist to contain and which is better not to create at all.
