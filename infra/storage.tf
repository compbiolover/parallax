# The run's own state: the SQLite database, the eMFD lexicon, and the exported
# dashboard payload.
#
# The payload lives here and goes no further. A CloudFront distribution would
# give it a public *.cloudfront.net URL, and `latest.js` at a public URL is a
# dated, running record of one person's news consumption — which digest/README.md
# refuses at some length, and correctly. The email is the delivery mechanism.

resource "aws_s3_bucket" "data" {
  bucket = var.data_bucket_name
}

# The free backup. An overwritten or truncated database is recoverable only if
# the previous version still exists, and the run overwrites it daily.
resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  # A version of the database per day, each hundreds of megabytes, kept forever
  # is a bill that grows without anyone deciding it should.
  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    # boto3 uploads the database with multipart. An interrupted upload leaves
    # parts that are billed and invisible until you look for them.
    abort_incomplete_multipart_upload {
      days_after_initiation = 3
    }
  }

  depends_on = [aws_s3_bucket_versioning.data]
}

# The single-writer lease. Not defensive: the datastore has no WAL, no
# busy_timeout past sqlite3's five-second default, and Datastore.__init__ runs
# the schema script and a migration pass on *every* open — so two processes
# opening the store while a migration is pending is the sharpest edge in the
# codebase. Step Functions sequences the scheduled path; this covers a manual
# run colliding with it.
resource "aws_dynamodb_table" "lease" {
  name         = "${var.name}-lease"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "lock_id"

  attribute {
    name = "lock_id"
    type = "S"
  }

  # Sweeps rows nobody will look at again. Correctness does not depend on it:
  # deploy/state.py compares expires_at inside the conditional write, because
  # TTL deletion is best-effort and documented as lagging up to 48 hours.
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}
