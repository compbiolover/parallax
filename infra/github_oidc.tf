# Lets .github/workflows/images.yml push to ECR with no stored key: GitHub mints
# a short-lived OIDC token, AWS trades it for temporary credentials, and there is
# no long-lived secret in the repository to leak or rotate.

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]

  # No thumbprint_list on purpose. Pinning one used to be mandatory and is now a
  # liability: AWS secures this endpoint against its own trust store, so a
  # hard-coded certificate thumbprint buys nothing and silently becomes a
  # time bomb the day GitHub rotates its chain — OIDC starts failing and the
  # cause is a constant nobody has looked at in a year. Omitted, AWS resolves it.
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Scoped to this repository. Without this condition the role is assumable
    # from *any* GitHub Actions workflow anywhere, which is a well-known way to
    # hand strangers a push credential.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:*"]
    }
  }
}

resource "aws_iam_role" "github_ecr_push" {
  name               = "${var.name}-github-ecr-push"
  description        = "Assumed by GitHub Actions to push images. Push only; deploys nothing."
  assume_role_policy = data.aws_iam_policy_document.github_assume.json
}

data "aws_iam_policy_document" "github_ecr_push" {
  statement {
    sid       = "Authenticate"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # this action does not take a resource
  }

  statement {
    sid = "Push"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
      # Pull as well, so a build can reuse cached layers it already pushed.
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [for r in aws_ecr_repository.image : r.arn]
  }
}

resource "aws_iam_role_policy" "github_ecr_push" {
  name   = "${var.name}-github-ecr-push"
  role   = aws_iam_role.github_ecr_push.id
  policy = data.aws_iam_policy_document.github_ecr_push.json
}
