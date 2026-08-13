locals {
  images = {
    core   = "cluster, summarize, snapshot, export, digest"
    scored = "ingest, backfill"
  }
}

resource "aws_ecr_repository" "image" {
  for_each = local.images

  name                 = "${var.name}-${each.key}"
  image_tag_mutability = "MUTABLE"

  # Catches a vulnerable base image without running anything extra.
  image_scanning_configuration {
    scan_on_push = true
  }

  tags = { Steps = each.value }
}

# `scored` is ~2.5 GB. Without this, every build's untagged layers stay forever
# and storage becomes the largest line on an otherwise small bill.
resource "aws_ecr_lifecycle_policy" "image" {
  for_each   = aws_ecr_repository.image
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last 10 tagged images"
        selection = {
          tagStatus      = "tagged"
          tagPatternList = ["*"]
          countType      = "imageCountMoreThan"
          countNumber    = 10
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Drop untagged layers after a day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      },
    ]
  })
}
