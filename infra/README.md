# Infrastructure

The daily run on AWS: two Fargate tasks in sequence, a SQLite database that
lives in S3 between runs, and a digest that arrives by email. `us-east-1`,
Terraform, one account.

Nothing here is multi-tenant. It is the same single-user run that has been
happening on a Mac, moved somewhere that does not have to be awake at 6am.

## Apply it, in this order

The order matters — several steps depend on the one before, and two of them
cannot be automated at all.

**1. Bootstrap the state backend.** Terraform's own state has to live somewhere
that Terraform is not yet managing.

```sh
cd infra/bootstrap
terraform init
terraform apply -var 'state_bucket_name=parallax-tfstate-CHANGE-ME'
terraform output backend_config     # paste into ../versions.tf, uncommented
```

**2. Apply the main config.**

```sh
cd infra
cp terraform.tfvars.example terraform.tfvars   # then edit it
terraform init      # picks up the backend you just pasted in
terraform apply
```

**3. Put the secret values in, by hand.** Terraform creates the secret container
and never its contents: anything Terraform knows ends up in state in plaintext,
and the point of a secret store is that these do not.

```sh
cat > runtime.json <<'JSON'
{
  "ANTHROPIC_API_KEY": "sk-ant-...",
  "PARALLAX_SMTP_HOST": "smtp.example.com",
  "PARALLAX_SMTP_PORT": "587",
  "PARALLAX_SMTP_USER": "you@example.com",
  "PARALLAX_SMTP_PASSWORD": "...",
  "PARALLAX_DIGEST_TO": "you+parallax-aws@example.com",
  "PARALLAX_DIGEST_FROM": "you@example.com",
  "PARALLAX_SMTP_STARTTLS": "1"
}
JSON
aws secretsmanager put-secret-value --secret-id parallax/runtime \
  --secret-string file://runtime.json
rm runtime.json
```

Note `PARALLAX_DIGEST_TO`. While the Mac job is still running, the AWS digest
goes to a **tagged alias** so both briefs arrive each morning and can be read
side by side. That comparison is the entire point of running in parallel: a
divergence in the numbers is far easier to see in two emails than by diffing two
SQLite files, and until they agree there is no evidence the migration preserved
anything.

**4. Upload the eMFD lexicon.** It is gitignored, so it is not in the image and
not in the repository.

```sh
terraform output -raw lexicon_upload_command | sh
```

Skipping this does not fail. `build_lexicon` warns and scores with the built-in
demo seed, which is illustrative only — so the run produces a complete,
plausible, fully-populated set of numbers from an instrument that was never
validated. The ingest task sets `PARALLAX_REQUIRE_LEXICON=1` to turn that into a
refusal instead, which is why step 4 comes before step 7.

**5. Push the images.** The GitHub OIDC role means Actions can do this with no
stored key — set `AWS_ROLE_ARN` in the repository's Actions variables from
`terraform output github_actions_role_arn`.

**6. Confirm the SNS subscription** from the email AWS sends. Terraform shows it
as `pending_confirmation` until you do; that is not a bug, it cannot be
confirmed programmatically by design. Until it is confirmed, every alarm here is
silent.

**7. Trigger one run by hand and watch it.**

```sh
terraform output -raw trigger_command | sh
```

Only then flip `schedule_enabled = true`.

## What to check on that first run

Three things, and the third is the one that needs deliberate checking because it
is the only one that fails silently.

```sh
terraform output -raw database_download_command | sh
python -m compare.history --db data/aws-parallax.sqlite
```

1. **A `snapshots` row for today.** The listing above shows it.
2. **A digest in the inbox.** Requires `digest` to be named in the aggregate
   task's command — it is deliberately absent from `DEFAULT_STEPS`, so a
   pipeline missing it does everything except the thing it exists to do, and
   reports success.
3. **The `meta` table's `lexicon` key naming the real eMFD file**, not the seed:

```sh
sqlite3 data/aws-parallax.sqlite "SELECT value FROM meta WHERE key='lexicon'"
```

Then break something on purpose — point a task definition at a tag that does not
exist — and confirm the alarm email actually arrives. An alarm nobody has seen
fire is an alarm nobody knows is wired up.

## Decisions worth knowing before you change something

**Public subnets, no NAT gateway.** A NAT gateway is ~$32/month before a byte
moves through it, a quarter of the running cost, and its only job would be to
hide tasks that accept no inbound connections. Instead: a public IP, and a
security group with zero ingress rules.

The corollary bites people: with **no** public IP and **no** NAT, a Fargate task
cannot reach ECR and dies at image pull with an error that never mentions
networking. `AssignPublicIp = "ENABLED"` in the state machine is load-bearing.

**Two IAM roles that are not interchangeable.** The *execution* role is what ECS
uses before your code runs — pull the image, fetch secrets, open a log stream.
The *task* role is what the process inside uses. Grant S3 to the execution role
and the task starts cleanly and then cannot read its own database.

The task role includes `s3:AbortMultipartUpload`, which looks like padding and
is not: boto3's `upload_file` switches to multipart above 8 MB and the database
is hundreds of megabytes, so a `PutObject`-only policy passes every test against
a small fixture and fails against the real store.

**No automatic retry on application failure.** `daily/__main__.py` returns 1 for
a partial failure, a total failure and a pre-loop crash alike, and 2 for a
malformed command. A retry policy cannot tell "GDELT was down" from "this
invocation can never work", so it would repeat the latter until the timeout.
Only ECS infrastructure faults retry; everything else alarms.

**The lease is not belt-and-braces.** The datastore has no WAL, no
`busy_timeout` past sqlite3's five-second default, and `Datastore.__init__` runs
the schema script and a migration pass on *every* open — two processes opening
the store while a migration is pending is the sharpest edge in the codebase.
Step Functions sequences the scheduled path; the DynamoDB lease is what stops a
manual run colliding with it.

**Timeouts must stay above ~90 minutes.** `scoring/liberty.py` polls the
Anthropic Batch API synchronously for up to an hour, so a shorter ceiling kills
legitimate runs rather than runaway ones. There is a variable validation rule
enforcing this.

## What is deliberately absent

- **No SES**, because there is no domain to verify yet. `digest/send.py` already
  does authenticated STARTTLS against whatever SMTP host you use, and Fargate
  egress on 587 is fine (AWS blocks 25, not 587).
- **No CloudFront, no public dashboard.** A distribution gets a public
  `*.cloudfront.net` URL, and `latest.js` there is a dated, running record of one
  person's news consumption at a URL — which `digest/README.md` refuses, and
  correctly. The payload is exported to a private prefix; the email is the
  delivery mechanism.
- **No podcast task.** Transcription runs in hours and is the largest single line
  of the bill. It stays on the workstation until it earns its own image.
- **No Postgres.** `datastore.kind` and `PARALLAX_POSTGRES_DSN` are documented in
  `settings.example.yaml` with no implementing code. One writer needs none of it.

## Teardown

`terraform destroy` in `infra/`, then `infra/bootstrap/`. Both state buckets and
the lock table carry `prevent_destroy`, so those need the lifecycle block
removed first — deliberately, because losing state means Terraform no longer
knows what it created.
